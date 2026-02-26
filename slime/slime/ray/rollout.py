import itertools
import logging
import multiprocessing
import os
import random
import time
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.rollout.base_types import call_rollout_fn
from slime.utils import logging_utils
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info, init_http_client
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.metric_utils import (
    MetricChecker,
    compute_pass_rate,
    compute_rollout_step,
    compute_statistics,
    dict_add_prefix,
)
from slime.utils.misc import Box, group_by, load_function
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slime.utils.types import Sample

from ..utils.metric_utils import has_repetition
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg, prm_pg=None):
        configure_logger()

        self.args = args
        self.pg = pg
        self.prm_pg = prm_pg
        _start_router(args, router_ip_attr="sglang_router_ip", router_port_attr="sglang_router_port")
        if self.args.prm_enable and self.args.prm_num_gpus > 0:
            _start_router(args, router_ip_attr="prm_router_ip", router_port_attr="prm_router_port")
        # TODO make args immutable
        init_tracking(args, primary=False, router_addr=f"http://{args.sglang_router_ip}:{args.sglang_router_port}")
        init_http_client(args)

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        self.custom_convert_samples_to_train_data_func = None
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only:
            self.all_rollout_engines = []
        else:
            num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
            num_engines = args.rollout_num_gpus // num_gpu_per_engine
            self.all_rollout_engines = [None] * num_engines
        self.num_new_engines = init_rollout_engines(args, pg, self.all_rollout_engines)
        if self.args.prm_enable and self.args.prm_num_gpus > 0:
            prm_num_gpu_per_engine = min(args.prm_num_gpus_per_engine, args.num_gpus_per_node)
            prm_num_engines = args.prm_num_gpus // prm_num_gpu_per_engine
            self.all_prm_engines = [None] * prm_num_engines
            self.num_new_prm_engines = init_prm_engines(args, prm_pg, self.all_prm_engines)
        else:
            self.all_prm_engines = []
            self.num_new_prm_engines = 0
        self.nodes_per_engine = max(1, args.rollout_num_gpus_per_engine // args.num_gpus_per_node)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self.rollout_id = -1

        self._metric_checker = MetricChecker.maybe_create(args)
        self._health_monitor = None
        if self.args.use_fault_tolerance:
            self._health_monitor = RolloutHealthMonitor(self, args)
            self._health_monitor.start()  # Start the monitor thread (in paused state)
            self._ci_fault_injection_pending = self.args.ci_test  # Flag for CI fault injection

    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if self.all_rollout_engines and self.all_rollout_engines[0]:
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self.all_rollout_engines[0].simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")

    def dispose(self):
        if self._metric_checker is not None:
            self._metric_checker.dispose()
        if self._health_monitor is not None:
            self._health_monitor.stop()

    # TODO maybe rename "rollout_engines" and "all_rollout_engines" later
    @property
    def rollout_engines(self):
        # when doing multi-node serving, we will only send request to node-0 for each engine.
        return self.all_rollout_engines[:: self.nodes_per_engine]

    def get_rollout_engines_and_lock(self, include_prm=False):
        engines = list(self.rollout_engines)
        num_new = self.num_new_engines
        if include_prm:
            prm_engines = [e for e in getattr(self, "all_prm_engines", []) if e is not None]
            engines.extend(prm_engines)
            num_new += getattr(self, "num_new_prm_engines", 0)
        return engines, self.rollout_engine_lock, num_new

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        start_time = time.time()
        self.rollout_id = rollout_id
        self.health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
        _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        data = self._convert_samples_to_train_data(data)
        return self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        self.health_monitoring_resume()

        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        data = result.data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        metrics = _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)
        if self._metric_checker is not None:
            self._metric_checker.on_eval(metrics)

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    def offload(self):
        self.health_monitoring_pause()
        return ray.get(
            [engine.release_memory_occupation.remote() for engine in self.rollout_engines if engine is not None]
        )

    def onload(self, tags: list[str] | None = None):
        return ray.get(
            [
                engine.resume_memory_occupation.remote(tags=tags)
                for engine in self.rollout_engines
                if engine is not None
            ]
        )

    def onload_weights(self):
        self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    def onload_kv(self):
        self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    def recover_rollout_engines(self):
        """Restart any dead rollout engines and update num_new_engines for update_weights detection."""
        self.health_monitoring_pause()
        if self.rollout_id == -1:
            return self.rollout_engines, self.rollout_engine_lock, self.num_new_engines

        dead_indices = [i for i, engine in enumerate(self.all_rollout_engines) if engine is None]
        self.num_new_engines = init_rollout_engines(self.args, self.pg, self.all_rollout_engines)
        logger.info(f"Recovered {self.num_new_engines} dead rollout engines")
        assert self.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
        if self.args.offload_rollout and dead_indices:
            new_engines = [self.all_rollout_engines[i] for i in dead_indices]
            ray.get([engine.release_memory_occupation.remote() for engine in new_engines])
            ray.get([engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]) for engine in new_engines])

        return self.rollout_engines, self.rollout_engine_lock, self.num_new_engines

    def clear_num_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear num_new_engines after update_weights
        self.num_new_engines = 0
        if hasattr(self, "num_new_prm_engines"):
            self.num_new_prm_engines = 0

    def health_monitoring_pause(self) -> None:
        if self._health_monitor is not None:
            self._health_monitor.pause()

    def health_monitoring_resume(self) -> None:
        if self._health_monitor is not None:
            self._health_monitor.resume()

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data = torch.load(
                self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                original_num_rows = len(data)
                rough_subsample_num_rows = int(original_num_rows * ratio)
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
                logger.info(
                    f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
                )
            metrics = None
        else:
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = data.metrics
            data = data.samples
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = list(itertools.chain.from_iterable(data))

            if not self.args.disable_rollout_trim_samples:
                global_batch_size = self.args.global_batch_size
                target_steps_per_rollout = getattr(self.args, "num_steps_per_rollout", None)
                # dynamic_history can expand one rollout into many step-wise samples.
                # In that case, honor num_steps_per_rollout by deriving a per-rollout
                # dynamic global batch size from the actual collected sample count.
                auto_dynamic_for_history = getattr(self.args, "dynamic_history", False) and target_steps_per_rollout is not None
                use_dynamic_gbs = self.args.use_dynamic_global_batch_size or auto_dynamic_for_history
                dynamic_target_steps = target_steps_per_rollout if auto_dynamic_for_history else None
                if use_dynamic_gbs:
                    logger.info(f"Collected {len(data)} samples from rollout to train with dynamic global batch size")
                    # TODO: this is a temporary solution, we should directly save dynamic_global_batch_size to rollout data
                    self._dynamic_global_batch_size = self._compute_dynamic_global_batch_size(
                        len(data), target_steps=dynamic_target_steps
                    )
                    global_batch_size = self._dynamic_global_batch_size

                if len(data) % global_batch_size != 0:
                    trim_len = (len(data) // global_batch_size) * global_batch_size
                    if trim_len == 0:
                        raise ValueError(f"Not enough samples {len(data)} for global_batch_size {global_batch_size}")
                    origin_data_length = len(data)
                    data = data[:trim_len]
                    logger.info(f"trim number of samples from {origin_data_length} to {trim_len}")
                logger.info(f"Final collected {len(data)} samples from rollout to train")

        return data, metrics

    def _compute_dynamic_global_batch_size(self, num_samples: int, target_steps: int | None = None) -> int:
        """Calculate dynamic global_batch_size from actual per-rollout samples.

        If target_steps is provided, choose global_batch_size to keep the realized
        number of training steps per rollout close to target_steps.
        Otherwise fallback to one-step behavior for backward compatibility.
        """
        dp_size = self.train_parallel_config["dp_size"]
        original_gbs = self.args.global_batch_size

        desired_steps = int(target_steps) if target_steps is not None and target_steps > 0 else 1
        # Target per-step samples, then round down to a multiple of dp_size.
        per_step_target = max(1, num_samples // desired_steps)
        dynamic_gbs = (per_step_target // dp_size) * dp_size

        if dynamic_gbs == 0:
            # Too few samples, use at least dp_size.
            dynamic_gbs = dp_size
            logger.warning(f"num_samples={num_samples} < dp_size={dp_size}, using dp_size as global_batch_size")

        realized_steps = max(1, num_samples // dynamic_gbs)
        # Calculate how many samples will be discarded after trim.
        wasted = num_samples % dynamic_gbs

        if dynamic_gbs != original_gbs or wasted > 0 or realized_steps != desired_steps:
            logger.info(
                f"Dynamic global_batch_size: {original_gbs} -> {dynamic_gbs} "
                f"(num_samples={num_samples}, dp_size={dp_size}, "
                f"target_steps={desired_steps}, realized_steps={realized_steps}, wasted={wasted})"
            )

        return dynamic_gbs

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        # TODO to be refactored (originally Buffer._set_data)
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            logger.info(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            # TODO may improve the format
            if evaluation:
                dump_data = dict(
                    samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
                )
            else:
                dump_data = dict(
                    samples=[sample.to_dict() for sample in data],
                )

            torch.save(dict(rollout_id=rollout_id, **dump_data), path)

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.rewards_normalization:
            if getattr(self.args, "dynamic_history", False):
                # dynamic_history + GRPO:
                # normalize one outcome per trajectory inside each task(group),
                # then broadcast that normalized value to all step samples in
                # the same trajectory.
                traj_reward_by_key: dict[tuple[int, int], float] = {}
                group_to_keys: dict[int, list[tuple[int, int]]] = {}
                key_by_sample: list[tuple[int, int]] = []
                for i, sample in enumerate(samples):
                    group_idx = int(sample.group_index) if sample.group_index is not None else -1
                    traj_idx = int(sample.index) if sample.index is not None else i
                    key = (group_idx, traj_idx)
                    key_by_sample.append(key)
                    if key not in traj_reward_by_key:
                        traj_reward_by_key[key] = float(raw_rewards[i])
                        group_to_keys.setdefault(group_idx, []).append(key)

                normalized_by_key: dict[tuple[int, int], float] = {}
                for _, keys in group_to_keys.items():
                    vals = torch.tensor([traj_reward_by_key[k] for k in keys], dtype=torch.float32)
                    vals = vals - vals.mean(dim=-1, keepdim=True)
                    if self.args.grpo_std_normalization:
                        if len(keys) > 1:
                            vals = vals / (vals.std(dim=-1, keepdim=True) + 1e-6)
                        else:
                            vals = torch.zeros_like(vals)
                    for j, key in enumerate(keys):
                        normalized_by_key[key] = float(vals[j].item())

                rewards = [normalized_by_key[key] for key in key_by_sample]
                return raw_rewards, rewards

            # non-dynamic_history + GRPO/GSPO:
            # normalize reward directly inside each task(group).
            group_to_indices: dict[int, list[int]] = {}
            for i, sample in enumerate(samples):
                group_idx = int(sample.group_index) if sample.group_index is not None else -1
                group_to_indices.setdefault(group_idx, []).append(i)

            rewards = list(raw_rewards)
            for _, idxs in group_to_indices.items():
                vals = torch.tensor([raw_rewards[i] for i in idxs], dtype=torch.float32)
                vals = vals - vals.mean(dim=-1, keepdim=True)
                if self.args.grpo_std_normalization:
                    if len(idxs) > 1:
                        vals = vals / (vals.std(dim=-1, keepdim=True) + 1e-6)
                    else:
                        vals = torch.zeros_like(vals)
                for j, sample_idx in enumerate(idxs):
                    rewards[sample_idx] = float(vals[j].item())
            return raw_rewards, rewards

        if self.args.advantage_estimator in ["reinforce_plus_plus_baseline"] and self.args.rewards_normalization:
            # group norm
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean

            if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)

            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def _drop_constant_reward_groups(self, samples: list[Sample]) -> list[Sample]:
        """Drop GRPO/GSPO groups whose rewards are all identical.

        Keep at least one group to avoid empty training data.
        """
        if not samples:
            return samples
        if self.args.advantage_estimator not in ["grpo", "gspo"] or not self.args.rewards_normalization:
            return samples

        raw_rewards = [float(sample.get_reward_value(self.args)) for sample in samples]
        group_to_indices: dict[int, list[int]] = {}
        for i, sample in enumerate(samples):
            group_idx = int(sample.group_index) if sample.group_index is not None else -1
            group_to_indices.setdefault(group_idx, []).append(i)

        constant_groups: list[int] = []
        for group_idx, idxs in group_to_indices.items():
            vals = [raw_rewards[i] for i in idxs]
            if len(vals) == 0:
                continue
            if max(vals) - min(vals) <= 1e-12:
                constant_groups.append(group_idx)

        if not constant_groups:
            return samples

        keep_groups = [g for g in group_to_indices.keys() if g not in set(constant_groups)]
        dropped_groups = list(constant_groups)
        if not keep_groups:
            # Keep one full group so batch is never empty.
            keep_group = next(iter(group_to_indices.keys()))
            keep_groups = [keep_group]
            dropped_groups = [g for g in constant_groups if g != keep_group]

        keep_set = set(keep_groups)
        filtered_samples = []
        for sample in samples:
            group_idx = int(sample.group_index) if sample.group_index is not None else -1
            if group_idx in keep_set:
                filtered_samples.append(sample)

        if len(filtered_samples) != len(samples):
            logger.warning(
                "Dropped constant-reward groups for %s: dropped=%s kept=%s samples %d -> %d",
                self.args.advantage_estimator,
                dropped_groups,
                keep_groups,
                len(samples),
                len(filtered_samples),
            )
        return filtered_samples

    def _post_process_step_wise_rewards(
        self, samples: list[Sample]
    ) -> tuple[list[list[float]], list[list[list[int]]], list[list[int]], list[int]]:
        """Build and normalize step-wise rewards metadata for training.

        Returns:
            step_wise_step_rewards, step_wise_step_token_spans, step_wise_step_indices, group_indices
        """
        step_wise_step_rewards: list[list[float]] = []
        step_wise_step_token_spans: list[list[list[int]]] = []
        step_wise_step_indices: list[list[int]] = []
        group_indices: list[int] = []

        for sample in samples:
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            step_wise_meta = metadata.get("step_wise", {}) if isinstance(metadata, dict) else {}
            if not isinstance(step_wise_meta, dict):
                step_wise_meta = {}

            # Preferred field for step_wise: reward for each step has already
            # been composed as (step_prm + outcome) in reward_func.
            raw_step_rewards = step_wise_meta.get("step_scores_with_outcome", None)
            if raw_step_rewards is None:
                # Backward-compatible fallback for old metadata.
                raw_step_rewards = step_wise_meta.get("step_scores", [])
            raw_step_spans = step_wise_meta.get("step_token_spans", [])
            raw_step_indices = step_wise_meta.get("step_indices", None)

            if isinstance(raw_step_rewards, (tuple, list)):
                step_rewards = [float(x) for x in raw_step_rewards]
            else:
                step_rewards = []
            if isinstance(raw_step_indices, (tuple, list)):
                step_indices = [int(x) for x in raw_step_indices]
            else:
                step_indices = list(range(len(step_rewards)))

            step_spans = []
            if isinstance(raw_step_spans, (tuple, list)):
                for span in raw_step_spans:
                    if (
                        isinstance(span, (tuple, list))
                        and len(span) == 2
                        and span[0] is not None
                        and span[1] is not None
                    ):
                        step_spans.append([int(span[0]), int(span[1])])

            # dynamic_history mode may intentionally omit explicit step spans.
            # In that case, infer one span from loss_mask (first/last 1).
            if len(step_spans) == 0 and getattr(self.args, "dynamic_history", False) and sample.loss_mask is not None:
                active_positions = [i for i, m in enumerate(sample.loss_mask) if int(m) == 1]
                if active_positions:
                    step_spans = [[active_positions[0], active_positions[-1] + 1]]
                    if len(step_rewards) == 0:
                        step_rewards = [float(sample.get_reward_value(self.args))]
                    if len(step_indices) == 0:
                        step_indices = [0]

            if not (len(step_rewards) == len(step_spans) == len(step_indices)):
                aligned_len = min(len(step_rewards), len(step_spans), len(step_indices))
                logger.warning(
                    "Step-wise metadata length mismatch for sample %s: rewards=%s spans=%s indices=%s, trim to %s",
                    sample.index,
                    len(step_rewards),
                    len(step_spans),
                    len(step_indices),
                    aligned_len,
                )
                step_rewards = step_rewards[:aligned_len]
                step_spans = step_spans[:aligned_len]
                step_indices = step_indices[:aligned_len]

            step_wise_step_rewards.append(step_rewards)
            step_wise_step_token_spans.append(step_spans)
            step_wise_step_indices.append(step_indices)
            group_indices.append(int(sample.group_index) if sample.group_index is not None else -1)

        # step_wise normalization is done in rollout for clarity:
        # normalize within same (task group, step_index) across trajectories.
        if self.args.rewards_normalization:
            stats: dict[tuple[int, int], tuple[float, float, int]] = {}
            for i, rewards_i in enumerate(step_wise_step_rewards):
                group_idx = int(group_indices[i])
                indices_i = step_wise_step_indices[i]
                aligned_len = min(len(rewards_i), len(indices_i))
                for pos in range(aligned_len):
                    key = (group_idx, int(indices_i[pos]))
                    v = float(rewards_i[pos])
                    sum_v, sum_sq_v, count_v = stats.get(key, (0.0, 0.0, 0))
                    stats[key] = (sum_v + v, sum_sq_v + v * v, count_v + 1)

            # Drop constant normalization groups (same reward in one
            # (group_index, step_index) bucket). Keep at least one group.
            all_keys = list(stats.keys())
            constant_keys: set[tuple[int, int]] = set()
            for key, (sum_v, sum_sq_v, count_v) in stats.items():
                if count_v <= 1:
                    # Keep single-sample buckets: no normalization, use raw reward.
                    continue
                mean_v = sum_v / count_v
                var_v = max(sum_sq_v / count_v - mean_v * mean_v, 0.0)
                if var_v <= 1e-12:
                    constant_keys.add(key)

            kept_keys = [k for k in all_keys if k not in constant_keys]
            dropped_keys = list(constant_keys)
            if not kept_keys and all_keys:
                keep_key = all_keys[0]
                kept_keys = [keep_key]
                dropped_keys = [k for k in dropped_keys if k != keep_key]

            kept_key_set = set(kept_keys)

            for i, rewards_i in enumerate(step_wise_step_rewards):
                group_idx = int(group_indices[i])
                indices_i = step_wise_step_indices[i]
                spans_i = step_wise_step_token_spans[i]
                aligned_len = min(len(rewards_i), len(indices_i))
                normalized_i = []
                filtered_indices_i = []
                filtered_spans_i = []
                for pos in range(aligned_len):
                    key = (group_idx, int(indices_i[pos]))
                    if key not in kept_key_set:
                        continue
                    sum_v, sum_sq_v, count_v = stats[key]
                    v = float(rewards_i[pos])
                    if count_v > 1:
                        mean_v = sum_v / count_v
                        var_v = max(sum_sq_v / count_v - mean_v * mean_v, 0.0)
                        std_v = var_v**0.5
                        v = (v - mean_v) / (std_v + 1e-6)
                    normalized_i.append(v)
                    filtered_indices_i.append(int(indices_i[pos]))
                    filtered_spans_i.append(spans_i[pos])
                step_wise_step_rewards[i] = normalized_i
                step_wise_step_indices[i] = filtered_indices_i
                step_wise_step_token_spans[i] = filtered_spans_i

                # If this sample loses all step entries, mark it as non-trainable.
                if len(normalized_i) == 0:
                    samples[i].remove_sample = True

            if dropped_keys:
                logger.warning(
                    "Dropped constant step_wise groups: dropped=%s kept=%s",
                    dropped_keys,
                    kept_keys,
                )

        return step_wise_step_rewards, step_wise_step_token_spans, step_wise_step_indices, group_indices

    def _convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
        """
        Convert inference generated samples to training data.
        """
        if self.custom_convert_samples_to_train_data_func is not None:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        samples = self._drop_constant_reward_groups(samples)
        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
        }
        if self.args.advantage_estimator in ["grpo", "gspo"]:
            train_data["group_indices"] = [int(sample.group_index) if sample.group_index is not None else -1 for sample in samples]

        if self.args.advantage_estimator == "step_wise":
            (
                step_wise_step_rewards,
                step_wise_step_token_spans,
                step_wise_step_indices,
                group_indices,
            ) = self._post_process_step_wise_rewards(samples)

            train_data["step_wise_step_rewards"] = step_wise_step_rewards
            train_data["step_wise_step_token_spans"] = step_wise_step_token_spans
            train_data["step_wise_step_indices"] = step_wise_step_indices
            train_data["group_indices"] = group_indices

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length

            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        # overwriting the raw reward
        if samples[0].metadata and "raw_reward" in samples[0].metadata:
            train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        if samples[0].multimodal_train_inputs is not None:
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        if "teacher_log_probs" in samples[0].__dict__:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        return train_data

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def _split_train_data_by_dp(self, data, dp_size):
        """Split the train data by data parallel size."""
        rollout_data = {}

        if "prompt" in data:
            rollout_data["prompt"] = data["prompt"]

        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths

        if self.args.balance_data:
            # Equal-size partitioning requires divisibility by dp_size.
            # Dynamic rollout/history can produce tail batches that violate this.
            use_equal_size = (len(total_lengths) % dp_size) == 0
            if not use_equal_size:
                logger.warning(
                    "balance-data fallback: num_samples=%d is not divisible by dp_size=%d; "
                    "using unequal-size seqlen balancing for this rollout step.",
                    len(total_lengths),
                    dp_size,
                )
            partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=use_equal_size)
        else:
            partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

        rollout_data_refs = []

        for i in range(dp_size):
            rollout_data = {}
            partition = partitions[i]
            rollout_data["partition"] = partition
            for key in [
                "tokens",
                "multimodal_train_inputs",
                "response_lengths",
                "rewards",
                "truncated",
                "loss_masks",
                "round_number",
                "sample_indices",
                "rollout_log_probs",
                "rollout_routed_experts",
                "prompt",
                "teacher_log_probs",
                "step_wise_step_rewards",
                "step_wise_step_token_spans",
                "step_wise_step_indices",
                "group_indices",
            ]:
                if key not in data:
                    continue
                val = [data[key][j] for j in partition]
                rollout_data[key] = val
            # keys that need to be splited at train side
            for key in [
                "raw_reward",
                "total_lengths",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = data[key]
            # Pass dynamic global_batch_size to training side
            if hasattr(self, "_dynamic_global_batch_size"):
                rollout_data["dynamic_global_batch_size"] = self._dynamic_global_batch_size
            rollout_data_refs.append(Box(ray.put(rollout_data)))
        return rollout_data_refs


def init_rollout_engines(args, pg, all_rollout_engines):
    if args.debug_train_only:
        return 0

    num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
    num_engines = args.rollout_num_gpus // num_gpu_per_engine
    assert len(all_rollout_engines) == num_engines
    if args.prefill_num_servers is not None:
        prefill_num_servers = args.prefill_num_servers * args.rollout_num_gpus_per_engine // num_gpu_per_engine
        assert (
            num_engines > prefill_num_servers
        ), f"num_engines {num_engines} should be larger than prefill_num_servers {prefill_num_servers}"

    pg, reordered_bundle_indices, reordered_gpu_ids = pg

    RolloutRayActor = ray.remote(SGLangEngine)

    rollout_engines = []
    for i in range(num_engines):
        if all_rollout_engines[i] is not None:
            continue

        num_gpus = 0.2
        num_cpus = num_gpus

        # Get the base GPU ID from placement group
        base_gpu_id = int(reordered_gpu_ids[i * num_gpu_per_engine])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpu_per_engine],
        )

        env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
            key: os.environ.get(key, default_val)
            for key, default_val in {
                "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
                "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
            }.items()
        }

        worker_type = "regular"
        if args.prefill_num_servers is not None:
            if i < prefill_num_servers:
                worker_type = "prefill"
            else:
                worker_type = "decode"

        rollout_engine = RolloutRayActor.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            scheduling_strategy=scheduling_strategy,
            runtime_env={
                "env_vars": env_vars,
            },
        ).remote(args, rank=i, worker_type=worker_type, base_gpu_id=base_gpu_id)

        rollout_engines.append((i, rollout_engine))
        all_rollout_engines[i] = rollout_engine

    num_new_engines = len(rollout_engines)

    if num_new_engines == 0:
        return num_new_engines

    if args.rollout_external:
        addr_and_ports = _allocate_rollout_engine_addr_and_ports_external(args=args, rollout_engines=rollout_engines)
    else:
        addr_and_ports = _allocate_rollout_engine_addr_and_ports_normal(
            args=args, num_engines=num_engines, rollout_engines=rollout_engines
        )

    # TODO: don't ray.get here to overlap train actor init with rollout engine init.
    # somehow if we don't sync here, the --debug-rollout-only mode will crash.
    init_handles = [engine.init.remote(**(addr_and_ports[rank])) for rank, engine in rollout_engines]
    ray.get(init_handles)

    return num_new_engines


def init_prm_engines(args, pg, all_prm_engines):
    if not args.prm_enable or args.prm_num_gpus <= 0:
        return 0
    assert pg is not None, "PRM placement group is required when PRM is enabled."

    num_gpu_per_engine = min(args.prm_num_gpus_per_engine, args.num_gpus_per_node)
    num_engines = args.prm_num_gpus // num_gpu_per_engine
    assert len(all_prm_engines) == num_engines

    pg, reordered_bundle_indices, reordered_gpu_ids = pg
    RolloutRayActor = ray.remote(SGLangEngine)

    prm_engines = []
    for i in range(num_engines):
        if all_prm_engines[i] is not None:
            continue

        num_gpus = 0.2
        num_cpus = num_gpus
        base_gpu_id = int(reordered_gpu_ids[i * num_gpu_per_engine])
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpu_per_engine],
        )

        env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
            key: os.environ.get(key, default_val)
            for key, default_val in {
                "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
                "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
            }.items()
        }

        prm_engine = RolloutRayActor.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": env_vars},
        ).remote(args, rank=i, worker_type="regular", base_gpu_id=base_gpu_id, engine_role="prm")

        prm_engines.append((i, prm_engine))
        all_prm_engines[i] = prm_engine

    num_new_engines = len(prm_engines)
    if num_new_engines == 0:
        return num_new_engines

    addr_and_ports = _allocate_prm_engine_addr_and_ports(
        args=args,
        num_engines=num_engines,
        prm_engines=prm_engines,
    )
    init_handles = [engine.init.remote(**(addr_and_ports[rank])) for rank, engine in prm_engines]
    ray.get(init_handles)
    return num_new_engines


def _allocate_rollout_engine_addr_and_ports_external(args, rollout_engines):
    addr_and_ports = []
    for rank, _ in rollout_engines:
        addr = args.rollout_external_engine_addrs[rank]
        [host, port] = addr.split(":")
        addr_and_ports.append(
            dict(
                dist_init_addr=addr,
                nccl_port=None,
                host=host,
                port=int(port),
            )
        )
    return addr_and_ports


def _allocate_prm_engine_addr_and_ports(*, args, num_engines, prm_engines):
    # mirror rollout allocator but use PRM engine parallel settings.
    num_engines_per_node = max(1, min(args.num_gpus_per_node, args.prm_num_gpus) // args.prm_num_gpus_per_engine)
    addr_and_ports = [{} for _ in range(num_engines)]

    visited_nodes = set()
    for rank, engine in prm_engines:
        if rank // num_engines_per_node in visited_nodes:
            continue
        visited_nodes.add(rank // num_engines_per_node)
        num_engines_on_this_node = num_engines_per_node - (rank % num_engines_per_node)

        def get_addr_and_ports(engine):
            start_port = 25000

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine)
        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports[current_rank]["host"] = get_addr()
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()

        if args.prm_num_gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = args.prm_num_gpus_per_engine // args.num_gpus_per_node
            if rank % num_node_per_engine == 0:
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    for i, _ in prm_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"PRM Engine {i} {key} is not set."
        logger.info(f"Ports for PRM engine {i}: {addr_and_ports[i]}")
    return addr_and_ports


def _allocate_rollout_engine_addr_and_ports_normal(*, args, num_engines, rollout_engines):
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    num_engines_per_node = max(
        1, min(args.num_gpus_per_node, args.rollout_num_gpus) // args.rollout_num_gpus_per_engine
    )
    addr_and_ports = [{} for _ in range(num_engines)]

    # Calculate prefill limit to identify prefill engines
    prefill_limit = 0
    if args.prefill_num_servers is not None:
        num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
        prefill_limit = args.prefill_num_servers * args.rollout_num_gpus_per_engine // num_gpu_per_engine

    visited_nodes = set()
    for rank, engine in rollout_engines:
        if rank // num_engines_per_node in visited_nodes:
            continue
        visited_nodes.add(rank // num_engines_per_node)
        # TODO: currently when restarting engines, we will set port for all engines on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting engine on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        num_engines_on_this_node = num_engines_per_node - (rank % num_engines_per_node)

        def get_addr_and_ports(engine):
            # use small ports to prevent ephemeral port between 32768 and 65536.
            # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
            start_port = 15000

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine)

        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports[current_rank]["host"] = get_addr()
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()

            if args.prefill_num_servers is not None and current_rank < prefill_limit:
                addr_and_ports[current_rank]["disaggregation_bootstrap_port"] = get_port()

        if args.rollout_num_gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = args.rollout_num_gpus_per_engine // args.num_gpus_per_node
            if rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports


def _start_router(args, router_ip_attr: str, router_port_attr: str):
    """Start a router for rollout or PRM engines."""
    if getattr(args, router_ip_attr, None) is not None:
        return

    setattr(args, router_ip_attr, _wrap_ipv6(get_host_info()[1]))
    if getattr(args, router_port_attr, None) is None:
        setattr(args, router_port_attr, find_available_port(random.randint(3000, 4000)))

    if args.use_slime_router:
        if router_ip_attr == "sglang_router_ip":
            assert args.prefill_num_servers is None, "slime router does not support prefill_num_servers."
        from slime.router.router import run_router

        router_args = copy(args)
        router_args.sglang_router_ip = getattr(args, router_ip_attr)
        router_args.sglang_router_port = getattr(args, router_port_attr)

    else:
        from sglang_router.launch_router import RouterArgs

        from slime.utils.http_utils import run_router

        router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
        router_args.host = getattr(args, router_ip_attr)
        router_args.port = getattr(args, router_port_attr)
        router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
        router_args.log_level = "warn"
        router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

        if router_ip_attr == "sglang_router_ip" and args.prefill_num_servers is not None:
            router_args.pd_disaggregation = True

        logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon
    process.start()
    # Wait 3 seconds
    time.sleep(3)
    assert process.is_alive()
    logger.info(f"Router launched at {getattr(args, router_ip_attr)}:{getattr(args, router_port_attr)}")


def _log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    if args.custom_eval_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if (samples := data[key].get("samples")) is not None:
            sample_metrics = compute_metrics_from_samples(args, samples)
            sample_metrics = {k: v for k, v in sample_metrics.items() if not k.startswith("zero_std/")}
            log_dict |= dict_add_prefix(sample_metrics, f"eval/{key}/")
            aborted = [s for s in samples if s.status == Sample.Status.ABORTED]
            log_dict[f"eval/{key}-aborted_ratio"] = len(aborted) / len(samples)
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")

    return log_dict


def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")


def compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()
    return log_dict


def compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if args.sglang_speculative_algorithm is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}
