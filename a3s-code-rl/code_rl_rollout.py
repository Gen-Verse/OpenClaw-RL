import asyncio
import atexit
import queue
import threading
import time

from code_rl_api_server import CodeRLAPIServer
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import eval_rollout
from slime.utils.async_utils import run
from slime.utils.types import Sample

_global_worker = None
_worker_lock = threading.Lock()


def get_global_worker(args, data_buffer):
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            _global_worker = AsyncRolloutWorker(args, data_buffer)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


class AsyncRolloutWorker:
    def __init__(self, args, data_buffer):
        self.args = args
        self.data_buffer = data_buffer
        self.running = True
        self.output_queue = queue.Queue(maxsize=100000)
        self.worker_thread = None
        self._submission_enabled = threading.Event()
        self._server = CodeRLAPIServer(
            args=args,
            output_queue=self.output_queue,
            submission_enabled=self._submission_enabled,
        )

    async def continuous_worker_loop(self):
        while self.running:
            await asyncio.sleep(1.0)

    def worker_thread_func(self):
        asyncio.run(self.continuous_worker_loop())

    def start(self):
        self._server.start()
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()

    def stop(self):
        self.running = False
        self._submission_enabled.clear()
        self._server.stop()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def pause_submission(self):
        if self._submission_enabled.is_set():
            self._submission_enabled.clear()
            discarded_groups = 0
            discarded_samples = 0
            while True:
                try:
                    _, group = self.output_queue.get_nowait()
                except queue.Empty:
                    break
                discarded_groups += 1
                discarded_samples += len(group)
            self._server.purge_record_files()
            if discarded_groups:
                print(
                    "[CodeRLWorker] submission paused; "
                    f"discarded {discarded_groups} queued groups / {discarded_samples} samples"
                )
            else:
                print("[CodeRLWorker] submission paused")

    def resume_submission(self):
        if not self._submission_enabled.is_set():
            self._submission_enabled.set()
            print("[CodeRLWorker] submission resumed")

    def get_completed_groups(self) -> list[tuple]:
        completed = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def get_queue_size(self) -> int:
        return self.output_queue.qsize()


async def _drain_output_queue(args, worker: AsyncRolloutWorker) -> list[list[Sample]]:
    target_data_size = args.rollout_batch_size
    data: list[list[Sample]] = []
    completed_groups: dict[int, list[Sample]] = {}
    start = time.time()
    last_progress = start

    while len(data) < target_data_size:
        completed = worker.get_completed_groups()
        if completed:
            last_progress = time.time()
            for group_id, group in completed:
                completed_groups[group_id] = group

        for group_id in list(completed_groups.keys()):
            if len(data) >= target_data_size:
                break
            group = completed_groups.pop(group_id)
            if any(sample.status == Sample.Status.ABORTED for sample in group):
                continue
            data.append(group)

        if time.time() - last_progress > 30:
            print(
                f"[CodeRLWorker] waiting for API-produced samples: {len(data)}/{target_data_size}, "
                f"queue={worker.get_queue_size()}",
                flush=True,
            )
            last_progress = time.time()

        if len(data) < target_data_size:
            await asyncio.sleep(0.05)

    data.sort(key=lambda group: group[0].index if group and group[0].index is not None else -1)
    print(f"[CodeRLWorker] drained {len(data)} groups in {time.time() - start:.2f}s", flush=True)
    return data


def generate_rollout_code_rl(args, rollout_id, data_buffer, evaluation=False):
    worker = get_global_worker(args, data_buffer)

    if evaluation:
        eval_output, _ = run(eval_rollout(args, rollout_id))
        return eval_output

    worker._server.reset_eval_scores()
    worker.resume_submission()
    completed_samples = run(_drain_output_queue(args, worker))
    worker.pause_submission()

    extra_metrics = None
    eval_scores = worker._server.drain_eval_scores()
    if eval_scores:
        extra_metrics = {"rollout/reward_mean": sum(eval_scores) / len(eval_scores)}
        print(
            f"[CodeRLWorker] reward_mean={extra_metrics['rollout/reward_mean']:.4f} "
            f"(n={len(eval_scores)})",
            flush=True,
        )

    return RolloutFnTrainOutput(samples=completed_samples, metrics=extra_metrics)


atexit.register(stop_global_worker)
