"""Preprocess a HuggingFace SWE dataset into slime-compatible JSONL format.

Output schema (one JSON object per line):
  {"text": "<problem_statement>", "metadata": {"instance": {...}, "data_source": "..."}}

Examples:
  # SWE-bench Verified (full)
  python preprocess_swe_dataset.py \
    --train-data-source SumanthRH/SWE-bench_Verified \
    --train-split test \
    --output-dir ~/data/swe_verified

  # SWE-Gym Subset — training set
  python preprocess_swe_dataset.py \
    --train-data-source SumanthRH/SWE-Gym-Subset \
    --train-split train \
    --output-dir ~/data/swe_gym_subset

  # SWE-Gym Subset — validation set (SWE-bench Verified test split)
  python preprocess_swe_dataset.py \
    --train-data-source SumanthRH/SWE-bench_Verified \
    --train-split test \
    --output-name validation.jsonl \
    --output-dir ~/data/swe_gym_subset

  # Smoke test: only first 5 records
  python preprocess_swe_dataset.py \
    --train-data-source SumanthRH/SWE-bench_Verified \
    --train-split test \
    --max-samples 5 \
    --output-dir ~/data/swe_verified_5
"""

import argparse
import json
import os

import datasets


def write_jsonl(dataset, out_path, data_source):
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset:
            record = {
                "text": example["problem_statement"],
                "metadata": {
                    "instance": {k: v for k, v in example.items()},
                    "data_source": data_source,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="~/data/swe_gym_subset")
    parser.add_argument("--output-name", default="train.jsonl")
    parser.add_argument("--train-data-source", default="SumanthRH/SWE-bench_Verified")
    parser.add_argument("--train-config", default="default")
    parser.add_argument("--train-split", default="test")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full split")
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ds = datasets.load_dataset(args.train_data_source, args.train_config)[args.train_split]
    if args.max_samples > 0:
        keep = min(args.max_samples, len(ds))
        ds = ds.select(range(keep))

    out_path = os.path.join(output_dir, args.output_name)
    write_jsonl(ds, out_path, args.train_data_source)

    print(f"Wrote {len(ds)} examples to {out_path}")
    print(f"train_data_source={args.train_data_source}, split={args.train_split}")


if __name__ == "__main__":
    main()
