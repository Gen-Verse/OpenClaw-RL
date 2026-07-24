from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

log = logging.getLogger("openclaw.puppet.workspace")

DEFAULTS = {
    "SOUL.md": "# SOUL.md\n\n_Who you are._\n",
    "AGENTS.md": "# AGENTS.md\n\n_Workspace rules._\n",
    "IDENTITY.md": "# IDENTITY.md\n\n_Fill in during first conversation._\n- **Name:**\n- **Emoji:**\n",
    "USER.md": "# USER.md\n\n_About your human._\n- **Name:**\n- **Timezone:**\n",
    "TOOLS.md": "# TOOLS.md\n\n_Local notes._\n",
    "HEARTBEAT.md": "# HEARTBEAT.md\n\n_Periodic tasks._\n",
}


class Workspace:
    def __init__(self, workspace: str) -> None:
        self.dir = Path(workspace)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "memory").mkdir(exist_ok=True)

    def init(self) -> list[str]:
        created: list[str] = []
        for name, default in DEFAULTS.items():
            p = self.dir / name
            if not p.exists():
                p.write_text(default)
                created.append(name)
        return created

    def read(self, name: str) -> str:
        p = self.dir / name
        return p.read_text() if p.exists() else ""

    def write(self, name: str, content: str) -> None:
        (self.dir / name).write_text(content)

    def set_field(self, name: str, field_name: str, value: str) -> None:
        lines = self.read(name).splitlines()
        marker = f"- **{field_name}:**"
        for i, line in enumerate(lines):
            if line.strip().startswith(marker):
                lines[i] = f"- **{field_name}:** {value}"
                self.write(name, "\n".join(lines) + "\n")
                return
        lines.append(f"- **{field_name}:** {value}")
        self.write(name, "\n".join(lines) + "\n")

    def list_files(self) -> list[dict]:
        return [
            {"name": f.name, "size": f.stat().st_size,
             "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()}
            for f in sorted(self.dir.iterdir()) if f.is_file() and not f.name.startswith(".")
        ]

    def identity(self) -> dict:
        return self._parse_fields("IDENTITY.md")

    def user(self) -> dict:
        return self._parse_fields("USER.md")

    def _parse_fields(self, name: str) -> dict:
        result: dict[str, str] = {}
        for line in self.read(name).splitlines():
            line = line.strip()
            if line.startswith("- **") and "**:" in line:
                key, _, val = line.partition("**:")
                key = key.replace("- **", "").strip().lower()
                val = val.strip()
                if val and not val.startswith("_"):
                    result[key] = val
        return result

    def backup(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = self.dir / f"backup_{ts}"
        dest.mkdir(exist_ok=True)
        for f in self.dir.iterdir():
            if f.is_file() and not f.name.startswith("."):
                shutil.copy2(f, dest / f.name)
        return str(dest)
