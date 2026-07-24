import os
import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional


class Memory:
    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)
        self.memory_dir = self.workspace / "memory"
        self.memory_file = self.workspace / "MEMORY.md"
        self.memory_dir.mkdir(exist_ok=True)

    def _today_file(self) -> Path:
        return self.memory_dir / f"{date.today().isoformat()}.md"

    def _append_daily(self, text: str) -> None:
        f = self._today_file()
        timestamp = datetime.now().strftime("%H:%M")
        entry = f"\n## [{timestamp}]\n{text}\n"
        with open(f, "a") as fh:
            fh.write(entry)

    def log(self, text: str, category: str = "general") -> None:
        tagged = f"[{category}] {text}" if category != "general" else text
        self._append_daily(tagged)

    def log_decision(self, decision: str, reason: str) -> None:
        self._append_daily(f"[decision] {decision}\n> Reason: {reason}")

    def log_action(self, action: str, result: str) -> None:
        self._append_daily(f"[action] {action}\n> Result: {result}")

    def log_error(self, error: str, context: str = "") -> None:
        ctx = f"\n> Context: {context}" if context else ""
        self._append_daily(f"[error] {error}{ctx}")

    def get_today(self) -> str:
        f = self._today_file()
        if f.exists():
            return f.read_text()
        return ""

    def get_recent(self, days: int = 3) -> str:
        entries = []
        for i in range(days):
            d = date.today() - __import__("datetime").timedelta(days=i)
            f = self.memory_dir / f"{d.isoformat()}.md"
            if f.exists():
                entries.append(f"### {d.isoformatformat()}\n{f.read_text()}")
        return "\n---\n".join(entries) if entries else "No recent memory."

    def get_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text()
        return ""

    def update_long_term(self, content: str) -> None:
        self.memory_file.write_text(content)

    def append_long_term(self, section: str, content: str) -> None:
        existing = self.get_long_term()
        marker = f"## {section}"
        if marker in existing:
            parts = existing.split(marker, 1)
            before = parts[0].rstrip()
            after = parts[1]
            next_heading = after.find("\n## ")
            if next_heading == -1:
                updated = f"{before}\n\n{marker}\n{content}\n{after}"
            else:
                updated = f"{before}\n\n{marker}\n{content}\n{after[next_heading:]}"
            self.memory_file.write_text(updated)
        else:
            with open(self.memory_file, "a") as f:
                f.write(f"\n\n{marker}\n{content}\n")

    def search(self, query: str, max_results: int = 10) -> list[dict]:
        results = []
        query_lower = query.lower()
        for f in sorted(self.memory_dir.glob("*.md"), reverse=True):
            if not results:
                content = f.read_text()
                for i, line in enumerate(content.splitlines(), 1):
                    if query_lower in line.lower():
                        results.append({"file": f.name, "line": i, "text": line.strip()})
                        if len(results) >= max_results:
                            return results
        long_term = self.get_long_term()
        for i, line in enumerate(long_term.splitlines(), 1):
            if query_lower in line.lower():
                results.append({"file": "MEMORY.md", "line": i, "text": line.strip()})
                if len(results) >= max_results:
                    break
        return results

    def list_files(self) -> list[dict]:
        files = []
        for f in sorted(self.memory_dir.glob("*.md"), reverse=True):
            stat = f.stat()
            files.append({
                "name": f.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        return files

    def consolidate(self) -> str:
        """Fold recent daily notes into MEMORY.md."""
        consolidated = []
        for f in sorted(self.memory_dir.glob("*.md")):
            content = f.read_text().strip()
            if content:
                consolidated.append(f"### {f.stem}\n{content}")
        if consolidated:
            header = "# Long-Term Memory\n\n_Auto-consolidated from daily notes._\n\n"
            self.memory_file.write_text(header + "\n\n---\n\n".join(consolidated))
            return f"Consolidated {len(consolidated)} daily files into MEMORY.md"
        return "Nothing to consolidate."

    def cleanup(self, keep_days: int = 30) -> int:
        """Remove daily files older than keep_days."""
        cutoff = time.time() - (keep_days * 86400)
        removed = 0
        for f in self.memory_dir.glob("*.md"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        return removed
