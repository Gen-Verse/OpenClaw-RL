#!/usr/bin/env python3
"""Aggregate WildClawBench ablation results across variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def extract_scores(summary: dict) -> dict:
    """Normalize whatever summary_all.json exposes into {category: score, overall: score}."""
    out = {}
    if not summary:
        return out

    # tolerate a few known shapes
    if isinstance(summary.get("categories"), dict):
        for cat, payload in summary["categories"].items():
            if isinstance(payload, dict):
                score = payload.get("overall_score", payload.get("score"))
                if score is not None:
                    out[cat] = float(score)
    for key in ("overall_score", "overall", "total_score"):
        if key in summary:
            out["overall"] = float(summary[key])
            break
    if "overall" not in out and out:
        out["overall"] = round(sum(out.values()) / len(out), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.results_dir)
    table = {}
    if not root.is_dir():
        root.mkdir(parents=True, exist_ok=True)
    for variant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        summary = load_summary(variant_dir / "summary_all.json")
        table[variant_dir.name] = extract_scores(summary)

    categories = sorted({cat for scores in table.values() for cat in scores if cat != "overall"})
    rows = []
    for variant, scores in table.items():
        row = {"variant": variant, "overall": scores.get("overall")}
        for cat in categories:
            row[cat] = scores.get(cat)
        rows.append(row)

    result = {"categories": categories, "rows": rows}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # also print a compact table
    header = ["variant", "overall", *categories]
    print("\t".join(header))
    for row in rows:
        print("\t".join(str(row.get(col, "")) for col in header))


if __name__ == "__main__":
    main()
