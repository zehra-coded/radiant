"""
Download script for the Rad-SpRL radiology report dataset.

This script is a PLANNER. It supports --dry-run, which prints what the
download WOULD do without contacting the network or writing any files.
The dataset is not downloaded in Stage 3.

Source:
    https://data.mendeley.com/datasets/yhb26hfz8n/1

Rad-SpRL is a Mendeley Data publication. Its actual annotation format
is NOT assumed by this script. Stage 3 will inspect the files if they
are present and document the format. If absent, the format remains
unverified and this is recorded as NOT RUN.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_DEST: Path = Path(__file__).resolve().parents[1] / "data" / "radsprl"

MENDELEY_PAGE: str = "https://data.mendeley.com/datasets/yhb26hfz8n/1"


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
    print(f"Source (Mendeley Data):   {MENDELEY_PAGE}")
    print(f"Destination directory:    {dest}")
    print()
    print("Planned steps:")
    print("  1. Create destination directory if missing.")
    print("  2. Open the Mendeley Data page in a browser.")
    print("  3. Download the dataset archive(s).")
    print("  4. Extract into the destination directory.")
    print("  5. Inspect annotation format (NOT ASSUMED).")
    print("  6. Document the mapping to the 14 NIH labels.")
    print()
    print("Stage 3 does NOT download the dataset. This run is a plan only.")
    print("RAD-SPRL DOWNLOAD: NOT RUN — deferred to a later stage.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dest = args.dest.resolve()

    if args.dry_run:
        dry_run_report(dest)
        return 0

    print(
        "Refusing to download without --dry-run. "
        "Rad-SpRL download is deferred. "
        "Set --dest and run with explicit intent in a later stage.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())