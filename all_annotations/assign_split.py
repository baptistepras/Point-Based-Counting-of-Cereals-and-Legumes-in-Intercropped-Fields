"""Batch-assign train/val/test splits to images currently marked "?" (unknown).

annotate.py leaves every newly-annotated image at split "?" (see its module
docstring) so annotating never blocks on deciding train/val/test up front.
Once you have enough "?" images, run this to assign the first --train of them
(sorted alphabetically) to train, the next --test to test, the next --val to
val; the remainder stays "?".

Runs on exactly one species (--wheat or --pea, required) at a time —
annotations_wheat.json and annotations_pea.json don't have the same number of
annotated images (most photos are single-species; only intercropped plots
appear in both), so a combined "?" list would mix two unrelated counts and
the resulting split sizes wouldn't match what you asked for in either file.
Since wheat and pea are trained as two entirely separate models on their own
test sets, there's no requirement that the two species agree on a shared
image's split.

Usage:
    python all_annotations/assign_split.py --wheat --train 50 --test 10 --val 10
    python all_annotations/assign_split.py --pea   --train 30 --test 5  --val 5
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE       = Path(__file__).resolve().parent
WHEAT_FILE = HERE / "annotations_wheat.json"
PEA_FILE   = HERE / "annotations_pea.json"


def main() -> None:
    """Read one species' annotation JSON, assign splits to "?" entries, write back."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheat", action="store_true")
    parser.add_argument("--pea",   action="store_true")
    parser.add_argument("--train", type=int, default=0)
    parser.add_argument("--test",  type=int, default=0)
    parser.add_argument("--val",   type=int, default=0)
    args = parser.parse_args()

    if args.wheat and args.pea:
        sys.exit("Error: --wheat and --pea are mutually exclusive")
    if not args.wheat and not args.pea:
        sys.exit("Error: select a species with --wheat or --pea")

    ann_file = WHEAT_FILE if args.wheat else PEA_FILE
    ann = json.loads(ann_file.read_text()) if ann_file.exists() else {}

    unknown = sorted(name for name, entry in ann.items() if entry.get("split", "?") == "?")

    total_asked = args.train + args.test + args.val
    if total_asked > len(unknown):
        print(f"WARNING: {total_asked} requested but only {len(unknown)} images with split '?'")

    i = 0
    for split, count in (("train", args.train), ("test", args.test), ("val", args.val)):
        for name in unknown[i : i + count]:
            ann[name]["split"] = split
        i += count

    ann_file.write_text(json.dumps(ann, indent=2))

    counts = Counter(v.get("split", "?") for v in ann.values())
    print(f"{ann_file.name}: splits updated ({len(ann)} annotated images):")
    for s in ("train", "val", "test", "?"):
        print(f"  {s:<6}: {counts.get(s, 0)}")


if __name__ == "__main__":
    main()
