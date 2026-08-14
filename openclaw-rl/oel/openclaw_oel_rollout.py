"""OEL rollout function for OpenClaw-RL.

Unified rollout supporting both online (OPCD-style) and offline (OEL-style)
experiential learning modes.

In online mode, behaves identically to the OPCD rollout.
In OEL modes (extract/deploy/consolidate), handles phase-specific data flow.
"""

import asyncio
import atexit
import queue
import threading
import time

from openclaw_oel_api_server import OpenClawOELAPIServer
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
        self._server = OpenClawOELAPIServer(
            args=args,
            output_queue=self.output_queue,
            submission_enabled=self._submission_enabled,
        )

    async def continuous_worker_loop(self):
        # Keepalive loop. Data production is request-driven in FastAPI handlers.
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
            self._server.purge_record_files()
            print("[OpenClawOELWorker] submission paused")

    def resume_submission(self):
        if not self._submission_enabled.is_set():
            self._submission_enabled.set()
            print("[OpenClawOELWorker] submission resumed")

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

    # Wait until we collect rollout_batch_size effective samples.
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
                f"[OpenClawOELWorker] waiting for OEL samples: {len(data)}/{target_data_size}, "
                f"queue={worker.get_queue_size()}",
                flush=True,
            )
            last_progress = time.time()

        if len(data) < target_data_size:
            await asyncio.sleep(0.05)

    data.sort(key=lambda group: group[0].index if group and group[0].index is not None else -1)
    print(f"[OpenClawOELWorker] drained {len(data)} groups in {time.time() - start:.2f}s", flush=True)
    return data


def generate_rollout_openclaw_oel(args, rollout_id, data_buffer, evaluation=False):
    """Main rollout entry point for OpenClaw OEL.

    Supports all OEL modes:
      - online: continuous OPCD-style training (wait for external requests)
      - extract: experience extraction only (no training samples produced)
      - deploy: trajectory collection only (no training samples produced)
      - consolidate: distillation training from pre-collected data + experience
    """
    worker = get_global_worker(args, data_buffer)

    if evaluation:
        eval_output, _ = run(eval_rollout(args, rollout_id))
        return eval_output

    # In extract/deploy modes, the server doesn't produce training samples.
    # The training loop still needs something — we signal to sleep and retry.
    mode = worker._server._mode
    if mode in (OpenClawOELAPIServer.MODE_EXTRACT, OpenClawOELAPIServer.MODE_DEPLOY):
        # These modes are inference-only. The server collects data but
        # doesn't submit training samples. Return empty to let the
        # training loop handle it (typically --val-only or similar flag).
        print(f"[OpenClawOELWorker] mode={mode}: inference-only, no training samples", flush=True)
        return RolloutFnTrainOutput(samples=[], metrics={"rollout/mode": 0.0})

    # online / consolidate modes: produce training samples
    worker._server.reset_eval_scores()
    worker.resume_submission()
    completed_samples = run(_drain_output_queue(args, worker))
    worker.pause_submission()

    extra_metrics = None
    eval_scores = worker._server.drain_eval_scores()
    if eval_scores:
        extra_metrics = {"rollout/prm_eval_score": sum(eval_scores) / len(eval_scores)}
        print(
            f"[OpenClawOELWorker] prm_eval_score={extra_metrics['rollout/prm_eval_score']:.4f} "
            f"(n={len(eval_scores)})",
            flush=True,
        )

    # Report experience stats
    exp_stats = worker._server.get_experience_stats()
    if exp_stats and extra_metrics is None:
        extra_metrics = {}
    if exp_stats:
        extra_metrics.update(exp_stats)

    return RolloutFnTrainOutput(samples=completed_samples, metrics=extra_metrics)


atexit.register(stop_global_worker)
