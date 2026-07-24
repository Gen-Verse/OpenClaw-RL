import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("openclaw.puppet.workspace")

REQUIRED_FILES = {
    "SOUL.md": "# SOUL.md\n\n_Who you are._\n",
    "AGENTS.md": "# AGENTS.md\n\n_Workspace rules._\n",
    "IDENTITY.md": "# IDENTITY.md\n\n_Fill in during first conversation._\n- **Name:**\n- **Emoji:**\n",
    "USER.md": "# USER.md\n\n_About your human._\n- **Name:**\n- **Timezone:**\n",
    "TOOLS.md": "# TOOLS.md\n\n_Local notes._\n",
    "HEARTBEAT.md": "# HEARTBEAT.md\n\n_Periodic tasks._\n",
}


class Workspace:
    def __init__(self, workspace_dir: str):
        self.dir = Path(workspace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir = self.dir / "memory"
        self.memory_dir.mkdir(exist_ok=True)

    def init(self) -> list[str]:
        created = []
        for name, default in REQUIRED_FILES.items():
            f = self.dir / name
            if not f.exists():
                f.write_text(default)
                created.append(name)
                logger.info("Created: %s", name)
        return created

    def read(self, filename: str) -> str:
        f = self.dir / filename
        if f.exists():
            return f.read_text()
        return ""

    def write(self, filename: str, content: str) -> None:
        f = self.dir / filename
        f.write_text(content)
        logger.info("Wrote: %s", filename)

    def update_field(self, filename: str, field_name: str, value: str) -> None:
        content = self.read(filename)
        marker = f"- **{field_name}:**"
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith(marker):
                lines[i] = f"- **{field_name}:** {value}"
                self.write(filename, "\n".join(lines) + "\n")
                return
        lines.append(f"- **{field_name}:** {value}")
        self.write(filename, "\n".join(lines) + "\n")

    def list_files(self) -> list[dict]:
        files = []
        for f in sorted(self.dir.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                stat = f.stat()
                files.append({
                    "name": f.name,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
        return files

    def get_identity(self) -> dict:
        content = self.read("IDENTITY.md")
        identity = {}
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- **") and "**:" in line:
                parts = line.split("**:", 1)
                if len(parts) == 2:
                    key = parts[0].replace("- **", "").strip().lower()
                    val = parts[1].strip()
                    if val and not val.startswith("_"):
                        identity[key] = val
        return identity

    def get_user(self) -> dict:
        content = self.read("USER.md")
        user = {}
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- **") and "**:" in line:
                parts = line.split("**:", 1)
                if len(parts) == 2:
                    key = parts[0].replace("- **", "").strip().lower()
                    val = parts[1].strip()
                    if val and not val.startswith("_"):
                        user[key] = val
        return user

    def backup(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.dir / f"backup_{ts}"
        backup_dir.mkdir(exist_ok=True)
        import shutil
        for f in self.dir.iterdir():
            if f.is_file() and not f.name.startswith("."):
                shutil.copy2(f, backup_dir / f.name)
        return str(backup_dir)
