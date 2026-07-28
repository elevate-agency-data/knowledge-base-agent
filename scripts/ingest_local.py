"""
Ingest a local atelier-style corpus into the ACTIVE brand's store.

The target DuckDB is the active profile's ``duckdb_path`` (shared/brand.py), so
select the brand via BRAND_PROFILE. Files map to indexes by folder domain, with
ligne_produit / zone / audience_role / matiere posed as metadata.

Usage:
    BRAND_PROFILE=hermes python -m scripts.ingest_local "C:/.../Data hermes/Rag_hermes"
    BRAND_PROFILE=hermes python -m scripts.ingest_local "<root>" --strategy fixed
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

from shared.brand import ACTIVE            # noqa: E402
from hybrid.config import DUCKDB_PATH      # noqa: E402
from hybrid.ingestion.local_ingest import ingest_local_tree  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest a local corpus tree.")
    ap.add_argument("root", help="Path to the corpus root (e.g. .../Rag_hermes)")
    ap.add_argument("--strategy", default="fixed",
                    choices=["fixed", "semantic", "hierarchical"])
    args = ap.parse_args()

    print(f"Brand      : {ACTIVE.name}")
    print(f"Store      : {DUCKDB_PATH}")
    print(f"Corpus root: {args.root}")
    print(f"Strategy   : {args.strategy}\n")

    res = ingest_local_tree(args.root, chunk_strategy=args.strategy)
    if res.get("status") != "success":
        print("ERROR:", res.get("message"))
        return 1

    print("\n── Summary ──")
    print(f"  indexes        : {len(res['indexes'])}")
    for idx in res["indexes"]:
        print(f"    {idx:22s} {res['chunks_per_index'][idx]} chunks")
    print(f"  files ingested : {res['files_ingested']}")
    print(f"  total chunks   : {res['total_chunks']}")
    if res["files_skipped"]:
        print(f"  skipped        : {len(res['files_skipped'])}")
        for s in res["files_skipped"][:10]:
            print(f"    - {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
