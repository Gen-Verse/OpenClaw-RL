from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("openclaw.puppet.skills")


class Skill:
    def __init__(self, name: str, path: Path, meta: dict) -> None:
        self.name = name
        self.path = path
        self.meta = meta
        self.description = meta.get("description", "")
        self.tools: dict[str, Callable] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        mod_file = self.path / "skill.py"
        if mod_file.exists():
            spec = importlib.util.spec_from_file_location(f"skill_{self.name}", mod_file)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "register"):
                    mod.register(self)
                self._loaded = True
                log.info("loaded skill: %s", self.name)

    def tool(self, name: str, func: Callable) -> None:
        self.tools[name] = func

    async def run(self, tool: str, **kw: Any) -> Any:
        if tool not in self.tools:
            raise ValueError(f"tool '{tool}' not in skill '{self.name}'")
        fn = self.tools[tool]
        import asyncio
        if asyncio.iscoroutinefunction(fn):
            return await fn(**kw)
        return fn(**kw)


class Registry:
    def __init__(self, skills_dir: str) -> None:
        self.dir = Path(skills_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.skills: dict[str, Skill] = {}
        self._scan()

    def _scan(self) -> None:
        for d in self.dir.iterdir():
            if d.is_dir():
                manifest = d / "skill.json"
                if manifest.exists():
                    try:
                        meta = json.loads(manifest.read_text())
                        self.skills[d.name] = Skill(d.name, d, meta)
                    except Exception:
                        log.exception("skip skill %s", d.name)

    def get(self, name: str) -> Skill | None:
        s = self.skills.get(name)
        if s:
            s.load()
        return s

    def list_all(self) -> list[dict]:
        return [
            {"name": s.name, "description": s.description,
             "tools": list(s.tools.keys()), "loaded": s._loaded}
            for s in self.skills.values()
        ]

    async def execute(self, skill: str, tool: str, **kw: Any) -> Any:
        s = self.get(skill)
        if not s:
            raise ValueError(f"skill '{skill}' not found")
        return await s.run(tool, **kw)

    def create(self, name: str, description: str, tools: dict[str, Callable] | None = None) -> Skill:
        d = self.dir / name
        d.mkdir(exist_ok=True)
        (d / "skill.json").write_text(json.dumps({"name": name, "description": description}, indent=2))
        s = Skill(name, d, {"name": name, "description": description})
        if tools:
            for tn, fn in tools.items():
                s.tool(tn, fn)
        self.skills[name] = s
        return s
