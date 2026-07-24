from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path


class Memory:
    def __init__(self, workspace: str) -> None:
        self.root = Path(workspace)
        self.dir = self.root / "memory"
        self.dir.mkdir(exist_ok=True)
        self.long_term = self.root / "MEMORY.md"

    def _today_path(self) -> Path:
        return self.dir / f"{date.today().isoformat()}.md"

    def _append(self, text: str, category: str = "general") -> None:
        ts = datetime.now().strftime("%H:%M")
        tag = f"[{category}] " if category != "general" else ""
        with open(self._today_path(), "a") as fh:
            fh.write(f"\n## [{ts}] {tag}{text}\n")

    def log(self, text: str, category: str = "general") -> None:
        self._append(text, category)

    def log_decision(self, what: str, why: str) -> None:
        self._append(f"{what}\n> Reason: {why}", "decision")

    def log_action(self, action: str, result: str) -> None:
        self._append(f"{action}\n> Result: {result}", "action")

    def log_error(self, error: str, ctx: str = "") -> None:
        extra = f"\n> Context: {ctx}" if ctx else ""
        self._append(f"{error}{extra}", "error")

    def today(self) -> str:
        p = self._today_path()
        return p.read_text() if p.exists() else ""

    def recent(self, days: int = 3) -> str:
        parts: list[str] = []
        for i in range(days):
            d = date.today() - timedelta(days=i)
            p = self.dir / f"{d.isoformat()}.md"
            if p.exists():
                parts.append(f"### {d}\n{p.read_text()}")
        return "\n---\n".join(parts) or "No recent memory."

    def long_term_content(self) -> str:
        return self.long_term.read_text() if self.long_term.exists() else ""

    def write_long_term(self, content: str) -> None:
        self.long_term.write_text(content)

    def append_long_term(self, section: str, content: str) -> None:
        existing = self.long_term_content()
        marker = f"## {section}"
        if marker in existing:
            idx = existing.index(marker)
            rest = existing[idx + len(marker):]
            next_h = rest.find("\n## ")
            tail = rest[next_h:] if next_h != -1 else ""
            updated = existing[:idx] + f"{marker}\n{content}\n" + tail
        else:
            updated = existing.rstrip() + f"\n\n{marker}\n{content}\n"
        self.long_term.write_text(updated)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        results: list[dict] = []
        q = query.lower()
        for f in sorted(self.dir.glob("*.md"), reverse=True):
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if q in line.lower():
                    results.append({"file": f.name, "line": i, "text": line.strip()})
                    if len(results) >= limit:
                        return results
        for i, line in enumerate(self.long_term_content().splitlines(), 1):
            if q in line.lower():
                results.append({"file": "MEMORY.md", "line": i, "text": line.strip()})
                if len(results) >= limit:
                    break
        return results

    def list_files(self) -> list[dict]:
        return [
            {"name": f.name, "size": f.stat().st_size,
             "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()}
            for f in sorted(self.dir.glob("*.md"), reverse=True)
        ]

    def consolidate(self) -> str:
        chunks: list[str] = []
        for f in sorted(self.dir.glob("*.md")):
            txt = f.read_text().strip()
            if txt:
                chunks.append(f"### {f.stem}\n{txt}")
        if chunks:
            self.long_term.write_text(
                "# Long-Term Memory\n\n_Auto-consolidated from daily notes._\n\n"
                + "\n\n---\n\n".join(chunks)
            )
            return f"Consolidated {len(chunks)} files."
        return "Nothing to consolidate."

    def cleanup(self, keep_days: int = 30) -> int:
        cutoff = time.time() - keep_days * 86400
        n = 0
        for f in self.dir.glob("*.md"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                n += 1
        return n
