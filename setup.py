#!/usr/bin/env python3
"""Setup script for OpenClaw Autonomous Controller."""
import json
import os
import shutil
from pathlib import Path


def setup():
    home = Path.home()
    openclaw_dir = home / ".openclaw"
    workspace_dir = openclaw_dir / "workspace"
    overrides_dir = Path(__file__).parent / "workspace-overrides"
    config_src = Path(__file__).parent / "autonomous" / "autonomous.json.example"
    config_dst = openclaw_dir / "autonomous.json"

    print("[1/4] Installing Python dependencies...")
    os.system("pip install -r autonomous/requirements.txt")

    print("[2/4] Copying workspace overrides...")
    for f in overrides_dir.glob("*"):
        dst = workspace_dir / f.name
        if dst.exists():
            backup = dst.with_suffix(dst.suffix + ".bak")
            shutil.copy2(dst, backup)
            print(f"  Backed up: {dst.name} -> {backup.name}")
        shutil.copy2(f, dst)
        print(f"  Copied: {f.name}")

    print("[3/4] Setting up config...")
    if not config_dst.exists():
        shutil.copy2(config_src, config_dst)
        print(f"  Created: {config_dst}")
    else:
        print(f"  Config exists: {config_dst}")

    print("[4/4] Creating directories...")
    (openclaw_dir / "tasks").mkdir(exist_ok=True)
    print("  Created: tasks/")

    print("\nSetup complete!")
    print(f"Config: {config_dst}")
    print(f"Workspace: {workspace_dir}")
    print("\nNext steps:")
    print("  1. Edit ~/.openclaw/autonomous.json with your gateway token")
    print("  2. Start the gateway: openclaw gateway run")
    print("  3. Run: python -m autonomous.run run")


if __name__ == "__main__":
    setup()
