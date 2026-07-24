#!/usr/bin/env python3
"""Setup script for OpenClaw Puppet."""
import os
import shutil
from pathlib import Path


def setup():
    home = Path.home()
    openclaw_dir = home / ".openclaw"
    workspace_dir = openclaw_dir / "workspace"
    overrides_dir = Path(__file__).parent / "workspace-overrides"
    config_src = Path(__file__).parent / "autonomous" / "puppet.json.example"
    config_dst = openclaw_dir / "puppet.json"

    print("[1/5] Installing Python dependencies...")
    os.system("pip install -r autonomous/requirements.txt")

    print("[2/5] Creating directories...")
    for d in ["tasks", "cron", "skills", "memory"]:
        (openclaw_dir / d).mkdir(exist_ok=True)
        print(f"  {d}/")

    print("[3/5] Copying workspace overrides...")
    for f in overrides_dir.glob("*"):
        dst = workspace_dir / f.name
        if dst.exists():
            backup = dst.with_suffix(dst.suffix + ".bak")
            shutil.copy2(dst, backup)
            print(f"  Backed up: {dst.name}")
        shutil.copy2(f, dst)
        print(f"  Copied: {f.name}")

    print("[4/5] Setting up config...")
    if not config_dst.exists():
        shutil.copy2(config_src, config_dst)
        print(f"  Created: {config_dst}")
    else:
        print(f"  Config exists: {config_dst}")

    print("[5/5] Initializing workspace...")
    from autonomous.workspace import Workspace
    ws = Workspace(str(workspace_dir))
    created = ws.init()
    if created:
        print(f"  Created: {', '.join(created)}")
    else:
        print("  All files exist.")

    print("\nDone. Edit ~/.openclaw/puppet.json with your gateway token.")
    print("Then: openclaw gateway run && python -m autonomous.run run")


if __name__ == "__main__":
    setup()
