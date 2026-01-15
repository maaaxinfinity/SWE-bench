#!/usr/bin/env python3
"""
Download SWE-bench datasets into the local Hugging Face cache or save to disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from datasets import DatasetDict, load_dataset


def normalize_dataset_name(name: str) -> str:
    lower = name.lower()
    if lower in {
        "swe-bench/swe-bench_oracle",
        "swe-bench_oracle",
        "swebench_oracle",
        "oracle",
    }:
        return "princeton-nlp/SWE-bench_oracle"
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SWE-bench datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "princeton-nlp/SWE-bench_oracle",
            "SWE-bench/SWE-bench_Verified",
        ],
        help="Dataset names to download",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Splits to download",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional output directory to save datasets to disk",
    )
    return parser.parse_args()


def load_split(name: str, split: str, cache_dir: Optional[str]):
    return load_dataset(name, split=split, cache_dir=cache_dir)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for raw_name in args.datasets:
        name = normalize_dataset_name(raw_name)
        ds_dict = DatasetDict()
        for split in args.splits:
            print(f"Downloading {name} [{split}]...")
            ds_dict[split] = load_split(name, split, args.cache_dir)
        print(f"Downloaded {name}: {', '.join(ds_dict.keys())}")

        if out_dir:
            safe_name = name.replace("/", "__")
            target = out_dir / safe_name
            ds_dict.save_to_disk(target)
            print(f"Saved to {target}")


if __name__ == "__main__":
    main()
