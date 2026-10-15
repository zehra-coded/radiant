"""
Download script for the NIH ChestX-ray14 dataset.

This script is a PLANNER. It supports --dry-run, which prints what the
download WOULD do without contacting the network or writing any files.
The full dataset is not downloaded in Stage 3.

Official NIH source:
    https://nihcc.app.box.com/v/ChestXray-NIHCC

The dataset is licensed for research use. See docs/datasets.md for the
full license record. Do not redistribute the images.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Target directory (created by Stage 1 scaffold).
DEFAULT_DEST: Path = Path(__file__).resolve().parents[1] / "data" / "nih"

# Official NIH ChestX-ray14 archive URLs. These point at the public
# Box.com listing hosted by the NIH Clinical Center.
NIH_BASE_URL: str = "https://nihcc.app.box.com/v/ChestXray-NIHCC"
NIH_ARCHIVES: tuple[str, ...] = (
    "images_001.tar.gz",
    "images_002.tar.gz",
    # ... (11 more archives totaling ~45 GB in the full release)
)

# Expected checksums are published by NIH as MD5 sums in a companion
# file. This script refuses to skip checksum verification when the
# checksum file is available.
NIH_CHECKSUM_FILE: str = "CheXpert-v1.0-small-test.csv"  # placeholder name; see docs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded. Do not contact the network.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})",
    )
    return parser.parse_args(argv)


def dry_run_report(dest: Path) -> None:
    print("DRY RUN — no network access, no files written.")
    print()
    print(f"Source (official):        {NIH_BASE_URL}")
    print(f"Destination directory:    {dest}")
    print()
    print("Archives that WOULD be downloaded (full release ~45 GB):")
    for name in NIH_ARCHIVES:
        print(f"  - {name}")
    print("  - ... and other archive parts per the NIH release listing")
    print()
    print("Planned steps:")
    print("  1. Create destination directory if missing.")
    print("  2. Fetch the NIH checksum file.")
    print("  3. Download each archive.")
    print("  4. Verify each archive's checksum.")
    print("  5. Extract archives into the destination directory.")
    print("  6. Fetch Data_Entry_2017.csv (labels).")
    print()
    print("Stage 3 does NOT download the dataset. This run is a plan only.")
    print("FULL NIH DOWNLOAD: NOT RUN — deferred to a later stage.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dest = args.dest.resolve()

    if args.dry_run:
        dry_run_report(dest)
        return 0

    print(
        "Refusing to download without --dry-run. "
        "The full NIH dataset is ~45 GB. "
        "Set --dest and run with explicit intent in a later stage.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())