#!/usr/bin/env python3
"""Run one skill-evolution round over WCB train-split outputs.

Usage:
  python -m skill_evolve.run_round \
    --raw-dir wildclaw-ablation/results/collect/raw \
    --skills-dir wildclaw-ablation/skills \
    --report wildclaw-ablation/results/evolve_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skill_evolve import grouper, judge, sessions as session_mod
from skill_evolve.evolver import evolve_group
from skill_evolve.store import EvolvingSkillStore
from skill_evolve.verifier import verify


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()[:48] or "skill"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--skills-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    skills_root = Path(args.skills_dir)
    store = EvolvingSkillStore(skills_root)

    sessions = session_mod.build_sessions(args.raw_dir, skills_root)
    judged = judge.backfill_scores(sessions)
    groups = grouper.group_sessions(sessions)

    report = {
        "sessions": len(sessions),
        "llm_judged": judged,
        "groups": {k: len(v) for k, v in groups.items()},
        "decisions": [],
    }

    for name, group in sorted(groups.items()):
        is_no_skill = name == grouper.NO_SKILL
        current = None if is_no_skill else store.current(name)
        history = [] if is_no_skill else store.history(name)

        candidate = evolve_group(name, group, current, history)
        if candidate is None:
            report["decisions"].append({"group": name, "action": "skip"})
            continue

        gate = verify(candidate, group)
        if not gate["accepted"]:
            report["decisions"].append(
                {"group": name, "action": candidate["action"], "verifier": "reject",
                 "reason": gate["reason"]}
            )
            continue

        skill_name = name if not is_no_skill else _slug(f"wcb-recovery-{len(store.list_skills())+1}")
        version = store.publish(skill_name, candidate["skill_md"], candidate["evidence"])
        report["decisions"].append(
            {"group": name, "action": candidate["action"], "verifier": "accept",
             "skill": skill_name, "version": version, "reason": gate["reason"]}
        )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
