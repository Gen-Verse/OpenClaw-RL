import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking


def eval_only(args):
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    rollout_manager, _ = create_rollout_manager(args, pgs["rollout"], pgs.get("prm"))

    ray.get(rollout_manager.eval.remote(0))
    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    eval_only(args)
