from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    db = tmp_path / "tasks.json"
    return subprocess.run(
        [sys.executable, "-m", "taskboard.cli", "--db", str(db), *args],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def test_add_and_list(tmp_path: Path) -> None:
    add = run_cli(tmp_path, "add", "ship release notes")
    assert add.returncode == 0
    assert "added #1 ship release notes" in add.stdout

    show = run_cli(tmp_path, "list")
    assert show.returncode == 0
    assert "#1 [open] ship release notes" in show.stdout


def test_done_marks_task_complete(tmp_path: Path) -> None:
    assert run_cli(tmp_path, "add", "clean docs").returncode == 0

    done = run_cli(tmp_path, "done", "1")
    assert done.returncode == 0
    assert "completed #1 clean docs" in done.stdout

    show = run_cli(tmp_path, "list")
    assert "#1 [done] clean docs" in show.stdout


def test_done_missing_id_returns_error(tmp_path: Path) -> None:
    missing = run_cli(tmp_path, "done", "99")
    assert missing.returncode == 1
    assert "task 99 not found" in missing.stderr


def test_add_rejects_blank_title(tmp_path: Path) -> None:
    add = run_cli(tmp_path, "add", "   ")
    assert add.returncode == 1
    assert "task title must not be empty" in add.stderr

    show = run_cli(tmp_path, "list")
    assert show.returncode == 0
    assert "no tasks" in show.stdout
