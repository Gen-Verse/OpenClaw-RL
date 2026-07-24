#!/usr/bin/env python3
"""Setup OpenClaw Puppet."""
import os
import shutil
from pathlib import Path

BASE = Path.home() / ".openclaw"
OVERRIDES = Path(__file__).parent / "workspace-overrides"
CONFIG_SRC = Path(__file__).parent / "autonomous" / "puppet.json.example"
CONFIG_DST = BASE / "puppet.json"


def main() -> None:
    print("[1/4] dependencies")
    os.system("pip install -r autonomous/requirements.txt")

    print("[2/4] directories")
    for d in ("tasks", "cron", "skills", "memory"):
        (BASE / d).mkdir(exist_ok=True)

    print("[3/4] workspace")
    ws_dir = BASE / "workspace"
    for f in OVERRIDES.glob("*"):
        dst = ws_dir / f.name
        if dst.exists():
            shutil.copy2(dst, dst.with_suffix(dst.suffix + ".bak"))
        shutil.copy2(f, dst)

    print("[4/4] config")
    if not CONFIG_DST.exists():
        shutil.copy2(CONFIG_SRC, CONFIG_DST)

    print(f"\ndone. edit {CONFIG_DST} then:")
    print("  openclaw gateway run")
    print("  python -m autonomous.run run")


if __name__ == "__main__":
    main()
