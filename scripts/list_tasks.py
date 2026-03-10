#!/usr/bin/env python3

"""List Gym task IDs defined in this repository's local tasks package."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import gymnasium as gym


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = PROJECT_ROOT / "tasks"


def _list_registered_tasks(prefix: str | None) -> list[str]:
    """Load local tasks package and list registered Gym IDs."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # Import side-effect: registers local tasks through gym.register(...)
    import tasks  # noqa: F401

    task_ids: list[str] = []
    for env_id, spec in gym.registry.items():
        if prefix and not env_id.startswith(prefix):
            continue
        entry_point = str(getattr(spec, "entry_point", ""))
        if "tasks." in entry_point or entry_point.startswith("tasks"):
            task_ids.append(env_id)
    return sorted(set(task_ids))


def _list_static_tasks(prefix: str | None) -> list[str]:
    """Parse gym.register(id=...) calls from tasks package __init__.py files."""
    task_ids: set[str] = set()
    for init_py in TASKS_ROOT.rglob("__init__.py"):
        try:
            tree = ast.parse(init_py.read_text(encoding="utf-8"), filename=str(init_py))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "register"):
                continue
            for kw in node.keywords:
                if kw.arg == "id" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    task_ids.add(kw.value.value)

    task_list = sorted(task_ids)
    if prefix:
        task_list = [task_id for task_id in task_list if task_id.startswith(prefix)]
    return task_list


def main() -> int:
    parser = argparse.ArgumentParser(description="List local Gym task IDs under this repository.")
    parser.add_argument(
        "--source",
        type=str,
        choices=("auto", "runtime", "static"),
        default="auto",
        help="Task discovery mode: runtime import, static scan, or auto fallback.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="Isaac-",
        help="Only print tasks with this prefix. Use empty string to disable filtering.",
    )
    args = parser.parse_args()

    prefix = args.prefix if args.prefix else None
    selected_source = args.source
    task_ids: list[str] = []

    if selected_source in ("runtime", "auto"):
        try:
            task_ids = _list_registered_tasks(prefix)
            selected_source = "runtime"
        except Exception as exc:
            if args.source == "runtime":
                print(f"[ERROR] Runtime listing failed: {exc}")
                return 1
            print(f"[WARN] Runtime listing failed ({exc}). Falling back to static scan.")

    if not task_ids and selected_source in ("static", "auto"):
        task_ids = _list_static_tasks(prefix)
        selected_source = "static"

    print(f"Found {len(task_ids)} tasks (source={selected_source}):")
    for task_id in task_ids:
        print(task_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
