"""
Ingest the ACTIVE brand's Drive atelier tree into its store.

Reads DRIVE_ROOT_FOLDER (from the active profile) via the service account and
projects the tree onto domain indexes + atelier metadata. Select the brand via
BRAND_PROFILE so the target DuckDB and Drive root both follow the profile.

Usage:
    BRAND_PROFILE=hermes python -m scripts.ingest_drive
    BRAND_PROFILE=hermes python -m scripts.ingest_drive --root Rag_hermes --max-files 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The Windows console defaults to cp1252, which cannot encode the box-drawing
# characters and accented file names this script prints — a crash there would
# lose the summary of an ingestion that actually succeeded.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.brand import ACTIVE                          # noqa: E402
from hybrid.config import DUCKDB_PATH, DRIVE_ROOT_FOLDER  # noqa: E402
from hybrid.tools.hybrid_add_data_atelier import hybrid_add_data_atelier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest the Drive atelier tree.")
    ap.add_argument("--root", default="", help="Drive root folder (default: brand's)")
    ap.add_argument("--strategy", default="fixed",
                    choices=["fixed", "semantic", "hierarchical"])
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()

    print(f"Brand      : {ACTIVE.name}")
    print(f"Store      : {DUCKDB_PATH}")
    print(f"Drive root : {args.root or DRIVE_ROOT_FOLDER}")
    print(f"Strategy   : {args.strategy}\n")

    res = hybrid_add_data_atelier(
        root_folder=args.root, chunk_strategy=args.strategy,
        max_files=args.max_files,
    )
    print("\n── Summary ──")
    print(f"  status         : {res.get('status')}")
    print(f"  {res.get('message','')}")
    for idx in res.get("indexes", []):
        print(f"    {idx:22s} {res['chunks_per_index'][idx]} chunks")
    print(f"  files ingested : {res.get('files_ingested', 0)}")
    print(f"  total chunks   : {res.get('total_chunks', 0)}")
    skipped = res.get("files_skipped", [])
    if skipped:
        print(f"  skipped        : {len(skipped)}")
        for s in skipped[:12]:
            print(f"    - {s.get('file')}: {s.get('reason')}")
    return 0 if res.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
