#!/usr/bin/env python3
"""Build a comparison table from terminal_report JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Name=path/to/terminal_report.json. Repeat for each variant.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for item in args.variant:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"invalid --variant value: {item!r}")
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append({"variant": name, **report})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
