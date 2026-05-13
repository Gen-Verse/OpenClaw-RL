from typing import Dict, Iterable


def validate_required_fields(event: Dict, required_fields: Iterable[str]) -> None:
    missing = [k for k in required_fields if k not in event]
    if missing:
        raise ValueError(f"missing required fields: {missing}")


def validate_action_type(event: Dict, allowed_action_types: Iterable[str]) -> None:
    if event.get("action_type") not in set(allowed_action_types):
        raise ValueError(f"invalid action_type: {event.get('action_type')}")


def validate_reward_components(event: Dict) -> None:
    rewards = event.get("reward_components", {})
    keys = {"pass", "quality", "safety", "human", "cost"}
    if set(rewards.keys()) != keys:
        raise ValueError(f"reward components mismatch: expect {keys}, got {set(rewards.keys())}")
