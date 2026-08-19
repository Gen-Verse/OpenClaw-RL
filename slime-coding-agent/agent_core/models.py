from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class RewardComponents:
    passed: float
    quality: float
    safety: float
    human: float
    cost: float

    @property
    def total(self) -> float:
        return self.passed + self.quality + self.safety + self.human - self.cost


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RolloutEvent:
    event_id: str
    timestamp: int
    repo_id: str
    task_id: str
    benchmark_id: str
    commit_base: str
    action_type: str
    action_payload: Dict[str, Any]
    command_results: List[Dict[str, Any]]
    test_results: Dict[str, Any]
    reward_components: Dict[str, float]
    total_reward: float
    final_status: str
    experiment: str
    usage: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
