"""Versioned skill store with per-skill history.

Layout:
  <root>/<name>/SKILL.md              current published version
  <root>/<name>/history/v{N}.md       previous versions
  <root>/<name>/history/v{N}_evidence.md  evidence that motivated v{N}
  <root>/manifest.json                name -> {version, updated_at}
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class EvolvingSkillStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"

    def _manifest(self) -> dict:
        if self.manifest_path.is_file():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    def _save_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def current(self, name: str) -> str | None:
        path = self.root / name / "SKILL.md"
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def history(self, name: str) -> list[dict]:
        history_dir = self.root / name / "history"
        if not history_dir.is_dir():
            return []
        entries = []
        for path in sorted(history_dir.glob("v*.md")):
            if path.name.endswith("_evidence.md"):
                continue
            evidence_path = path.with_name(path.stem + "_evidence.md")
            entries.append(
                {
                    "version": path.stem,
                    "content": path.read_text(encoding="utf-8"),
                    "evidence": evidence_path.read_text(encoding="utf-8")
                    if evidence_path.is_file()
                    else "",
                }
            )
        return entries

    def publish(self, name: str, skill_md: str, evidence: str) -> int:
        manifest = self._manifest()
        prev_version = int(manifest.get(name, {}).get("version", 0))
        version = prev_version + 1

        skill_dir = self.root / name
        history_dir = skill_dir / "history"
        history_dir.mkdir(parents=True, exist_ok=True)

        current_path = skill_dir / "SKILL.md"
        if current_path.is_file():
            shutil.copy2(current_path, history_dir / f"v{prev_version}.md")

        current_path.write_text(skill_md.rstrip() + "\n", encoding="utf-8")
        (history_dir / f"v{version}_evidence.md").write_text(
            evidence.rstrip() + "\n", encoding="utf-8"
        )

        manifest[name] = {"version": version, "updated_at": _now()}
        self._save_manifest(manifest)
        return version

    def list_skills(self) -> list[str]:
        return sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and (p / "SKILL.md").is_file()
        )
