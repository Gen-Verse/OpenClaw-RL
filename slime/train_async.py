import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.misc import Box
from slime.utils.misc import should_run_periodic_action


def _merge_partitioned_rollout_refs(rollout_data_refs):
    """Merge actor-DP rollout refs for single-DP PRM teacher passes."""
    if len(rollout_data_refs) <= 1:
        return rollout_data_refs

    parts = ray.get([ref.inner for ref in rollout_data_refs])
    total = None
    for part in parts:
        if "total_lengths" in part:
            total = len(part["total_lengths"])
            break
    if total is None:
        total = max(max(part["partition"]) for part in parts if part["partition"]) + 1

    merged = {"partition": list(range(total))}
    for part in parts:
        partition = list(part["partition"])
        for key, value in part.items():
            if key == "partition":
                continue
            if isinstance(value, list) and len(value) == len(partition):
                dest = merged.setdefault(key, [None] * total)
                for idx, item in zip(partition, value, strict=True):
                    dest[idx] = item
            elif key not in merged:
                merged[key] = value

    return [Box(ray.put(merged))]


def _merge_student_topk_payloads(payloads):
    """Merge one student top-k payload per actor DP rank into original order."""
    if len(payloads) <= 1:
        return payloads[0]

    total = None
    for payload in payloads:
        partition = payload.get("partition")
        if partition:
            total = max(total or 0, max(partition) + 1)
    if total is None:
        return payloads[0]

    merged = [None] * total
    seen_dp_ranks = set()
    for payload in payloads:
        dp_rank = payload.get("dp_rank")
        if dp_rank in seen_dp_ranks:
            continue
        partition = payload.get("partition")
        topk_indices = payload.get("topk_indices", [])
        if not partition or len(partition) != len(topk_indices):
            continue
        seen_dp_ranks.add(dp_rank)
        for idx, value in zip(partition, topk_indices, strict=True):
            merged[idx] = value

    if any(value is None for value in merged):
        missing = [idx for idx, value in enumerate(merged) if value is None]
        raise RuntimeError(f"Missing student top-k payload for sample indices: {missing[:8]}")
    return {"topk_indices": merged}


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"], pgs.get("prm"))

    # create the actor, critic, and (optionally) PRM teacher models
    actor_model, critic_model, prm_teacher_model = create_training_models(args, pgs, rollout_manager)

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if prm_teacher_model is not None:
            prm_teacher_rollout_data_ref = _merge_partitioned_rollout_refs(rollout_data_curr_ref)
            distill_topk = int(getattr(args, "distill_topk", 0) or 0)
            subset_mode = getattr(args, "distill_subset_mode", "student")
            if distill_topk > 0 and subset_mode == "student":
                # Three-pass dance for student-driven Sₜ:
                #   1. actor runs old_actor → student top-K indices (one payload
                #      per actor DP rank, merged below into original sample order).
                #   2. PRM teacher runs forward gathering at those indices.
                #   3. set on actor; actor.train re-runs old_actor with emit_topk
                #      (recomputes the same student top-K) and proceeds to train.
                student_topk_futures = actor_model.async_compute_student_topk(rollout_id, rollout_data_curr_ref)
                if len(rollout_data_curr_ref) <= 1:
                    student_topk = ray.get(student_topk_futures[0])
                else:
                    student_topk = _merge_student_topk_payloads(ray.get(student_topk_futures))
                prm_teacher_log_probs = ray.get(
                    prm_teacher_model.async_gather_at_indices(
                        rollout_id, prm_teacher_rollout_data_ref, student_topk["topk_indices"]
                    )[0]
                )
            else:
                prm_teacher_futures = prm_teacher_model.async_train(rollout_id, prm_teacher_rollout_data_ref)
                prm_teacher_log_probs = ray.get(prm_teacher_futures[0])
            actor_model.set_prm_teacher_log_probs(prm_teacher_log_probs)

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
            if args.use_critic:
                critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            actor_model.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
