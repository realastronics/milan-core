"""
build_include_manifest.py

Pulls the OFFICIAL label, split, and INCLUDE-50-membership metadata for INCLUDE from
the ai4bharat/INCLUDE dataset on HuggingFace, and writes a flat manifest CSV that
extract_landmarks.py reads from. This exists specifically so label assignment and
train/val/test splitting are never guessed or reinvented locally — they come from
AI4Bharat's own metadata, matching label_path exactly as published:
https://huggingface.co/datasets/ai4bharat/INCLUDE

Usage:
    python model/data/build_include_manifest.py --output model/data/include_manifest.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def normalize_label(raw_label: str) -> str:
    """
    Matches AI4Bharat's own reference-implementation normalization exactly
    (see INCLUDE/dataset.py: `"".join([i for i in label if i.isalpha()]).lower()`),
    so results stay comparable to their baseline (T-A11).

    "50. Yellow" -> "yellow"
    "3. happy"   -> "happy"
    "Thank you"  -> "thankyou"
    """
    return re.sub(r"[^a-zA-Z]", "", raw_label).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local manifest of official INCLUDE labels/splits.")
    parser.add_argument("--output", type=Path, default=Path("model/data/include_manifest.csv"))
    args = parser.parse_args()

    rows = []
    for split in ["train", "validation", "test"]:
        # HF dataset script exposes this split as "val" in its own naming (per the dataset card);
        # try both, since HF `load_dataset` split names occasionally differ from the card's prose.
        try:
            ds = load_dataset("ai4bharat/INCLUDE", split=split)
        except ValueError:
            ds = load_dataset("ai4bharat/INCLUDE", split="val" if split == "validation" else split)

        for row in ds:
            rows.append(
                {
                    "video_path": row["video_path"],
                    "parent_label": row["parent_label"],
                    "label_raw": row["label"],
                    "label": normalize_label(row["label"]),
                    "include_50": bool(row["include_50"]),
                    "split": "val" if split == "validation" else split,
                }
            )

    manifest = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)

    print(f"Wrote {len(manifest)} rows to {args.output}")
    print(f"Splits: {manifest['split'].value_counts().to_dict()}")
    print(f"Unique labels: {manifest['label'].nunique()}")
    print(f"INCLUDE-50 rows: {int(manifest['include_50'].sum())}")


if __name__ == "__main__":
    main()
