import importlib
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("openclaw.puppet.skills")


class Skill:
    def __init__(self, name: str, path: Path, manifest: dict):
        self.name = name
        self.path = path
        self.manifest = manifest
        self.description = manifest.get("description", "")
        self.tools: dict[str, Callable] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        module_path = self.path / "skill.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location(f"skill_{self.name}", module_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "register"):
                    mod.register(self)
                self._loaded = True
                logger.info("Loaded skill: %s", self.name)

    def register_tool(self, name: str, func: Callable) -> None:
        self.tools[name] = func

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    async def run_tool(self, name: str, **kwargs) -> Any:
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found in skill '{self.name}'")
        func = self.tools[name]
        if callable(func):
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return await func(**kwargs)
            return func(**kwargs)
        return func


class SkillRegistry:
    def __init__(self, skills_dir: str):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(exist_ok=True)
        self.skills: dict[str, Skill] = {}
        self._scan()

    def _scan(self) -> None:
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                manifest_file = skill_dir / "skill.json"
                if manifest_file.exists():
                    try:
                        import json
                        manifest = json.loads(manifest_file.read_text())
                        skill = Skill(skill_dir.name, skill_dir, manifest)
                        self.skills[skill_dir.name] = skill
                        logger.info("Found skill: %s", skill_dir.name)
                    except Exception:
                        logger.exception("Failed to load skill: %s", skill_dir.name)

    def get(self, name: str) -> Skill | None:
        skill = self.skills.get(name)
        if skill:
            skill.load()
        return skill

    def list_skills(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "tools": list(s.tools.keys()),
                "loaded": s._loaded,
            }
            for s in self.skills.values()
        ]

    def has_skill(self, name: str) -> bool:
        return name in self.skills

    async def run_tool(self, skill_name: str, tool_name: str, **kwargs) -> Any:
        skill = self.get(skill_name)
        if not skill:
            raise ValueError(f"Skill '{skill_name}' not found")
        return await skill.run_tool(tool_name, **kwargs)

    def create_skill(self, name: str, description: str, tools: dict[str, Callable] | None = None) -> Skill:
        skill_dir = self.skills_dir / name
        skill_dir.mkdir(exist_ok=True)
        import json
        manifest = {"name": name, "description": description, "version": "1.0.0"}
        (skill_dir / "skill.json").write_text(json.dumps(manifest, indent=2))
        skill = Skill(name, skill_dir, manifest)
        if tools:
            for tool_name, func in tools.items():
                skill.register_tool(tool_name, func)
        self.skills[name] = skill
        return skill
