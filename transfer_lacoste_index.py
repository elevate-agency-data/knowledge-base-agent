"""
Transfer Lacoste indexes from an old DuckDB to the current one.

Usage:
    python transfer_lacoste_index.py --old PATH_TO_OLD_DB [--keyword lacoste]

Steps:
    1. Opens both DBs (old read-only, current read-write)
    2. Finds all tables in the old DB whose index_name contains the keyword
    3. Copies the chunk tables that don't already exist in the current DB
    4. Inserts missing entries into hybrid_indexes registry
    5. Rebuilds HNSW + FTS indexes for each transferred table

Stop Streamlit before running this script.
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Transfer Lacoste indexes between DuckDB files")
    parser.add_argument("--old",     required=True,        help="Path to the old .duckdb file")
    parser.add_argument("--new",     default="hybrid/data/hybrid.duckdb", help="Path to the current .duckdb file")
    parser.add_argument("--keyword", default="lacoste",    help="Filter: only transfer indexes whose name contains this keyword (case-insensitive)")
    parser.add_argument("--dry-run", action="store_true",  help="Show what would be transferred without doing it")
    args = parser.parse_args()

    try:
        import duckdb
    except ImportError:
        print("ERROR: duckdb not installed. Run: pip install duckdb")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  SOURCE  : {args.old}")
    print(f"  TARGET  : {args.new}")
    print(f"  KEYWORD : {args.keyword}")
    print(f"  DRY RUN : {args.dry_run}")
    print(f"{'='*60}\n")

    # ── Open old DB (read-only) ───────────────────────────────────────────────
    try:
        old_con = duckdb.connect(args.old, read_only=True)
    except Exception as e:
        print(f"ERROR: Cannot open old DB: {e}")
        sys.exit(1)

    # ── Find matching indexes in old DB ───────────────────────────────────────
    try:
        old_indexes = old_con.execute(
            "SELECT index_name, embedding_model, chunk_strategy FROM hybrid_indexes ORDER BY index_name"
        ).fetchall()
    except Exception as e:
        print(f"ERROR: Cannot read hybrid_indexes from old DB: {e}")
        old_con.close()
        sys.exit(1)

    keyword = args.keyword.lower()
    matching = [row for row in old_indexes if keyword in row[0].lower()]

    if not matching:
        print(f"No indexes found containing '{args.keyword}' in old DB.")
        print("\nAll indexes in old DB:")
        for row in old_indexes:
            print(f"  - {row[0]}")
        old_con.close()
        sys.exit(0)

    print(f"Found {len(matching)} index(es) matching '{args.keyword}':")
    for row in matching:
        index_name, emb_model, chunk_strat = row
        # Count chunks in old DB
        safe = _sanitize(index_name)
        table = f"chunks_{safe}"
        try:
            count = old_con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception:
            count = "?"
        print(f"  - {index_name:50s} | {count} chunks | model={emb_model} | strategy={chunk_strat}")

    if args.dry_run:
        print("\nDry run — nothing transferred.")
        old_con.close()
        return

    print("\nProceed with transfer? [y/N] ", end="")
    confirm = input().strip().lower()
    if confirm != "y":
        print("Aborted.")
        old_con.close()
        sys.exit(0)

    # ── Open current DB (read-write) ──────────────────────────────────────────
    try:
        new_con = duckdb.connect(args.new, config={"hnsw_enable_experimental_persistence": True})
    except Exception as e:
        print(f"ERROR: Cannot open current DB: {e}")
        old_con.close()
        sys.exit(1)

    # Load extensions
    for ext in ("vss", "fts"):
        try:
            new_con.execute(f"LOAD {ext}")
        except Exception:
            try:
                new_con.execute(f"INSTALL {ext}")
                new_con.execute(f"LOAD {ext}")
            except Exception:
                print(f"  WARNING: Could not load {ext} extension")

    # Get existing indexes in current DB
    existing = set(
        row[0] for row in new_con.execute(
            "SELECT index_name FROM hybrid_indexes"
        ).fetchall()
    )

    # ── Attach old DB ─────────────────────────────────────────────────────────
    new_con.execute(f"ATTACH '{args.old}' AS old_db (READ_ONLY)")

    transferred = 0
    skipped     = 0
    errors      = 0

    for index_name, emb_model, chunk_strat in matching:
        safe  = _sanitize(index_name)
        table = f"chunks_{safe}"

        print(f"\n[{index_name}]")

        if index_name in existing:
            print(f"  SKIP — already exists in current DB")
            skipped += 1
            continue

        # ── Copy table ────────────────────────────────────────────────────────
        try:
            # Check table exists in old DB
            old_tables = [
                r[0] for r in new_con.execute(
                    "SELECT table_name FROM duckdb_tables() "
                    "WHERE database_name='old_db' AND schema_name='main'"
                ).fetchall()
            ]
            if table not in old_tables:
                print(f"  ERROR — table '{table}' not found in old DB (skipping)")
                errors += 1
                continue

            chunk_count = new_con.execute(
                f"SELECT COUNT(*) FROM old_db.{table}"
            ).fetchone()[0]
            print(f"  Copying {chunk_count} chunks...")

            # Detect embedding dimension from the table
            dim = _detect_dim(new_con, table)
            print(f"  Detected embedding dimension: {dim}")

            # Create table in current DB
            new_con.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id              VARCHAR PRIMARY KEY,
                    index_name      VARCHAR,
                    content         TEXT,
                    embedding       FLOAT[{dim}],
                    source_url      VARCHAR,
                    file_name       VARCHAR,
                    file_type       VARCHAR,
                    created_at      DATE,
                    updated_at      DATE,
                    author          VARCHAR,
                    domaine         VARCHAR,
                    langue          VARCHAR,
                    tags            VARCHAR[],
                    chunk_index     INTEGER,
                    chunk_total     INTEGER,
                    chunk_strategy  VARCHAR,
                    parent_chunk_id VARCHAR,
                    embedding_model VARCHAR,
                    embedding_dim   INTEGER
                )
            """)

            # Copy data
            new_con.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM old_db.{table}")
            actual = new_con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  Copied: {actual} chunks")

            # Register in hybrid_indexes
            new_con.execute(
                "INSERT OR IGNORE INTO hybrid_indexes (index_name, embedding_model, chunk_strategy) VALUES (?, ?, ?)",
                [index_name, emb_model, chunk_strat]
            )
            print(f"  Registered in hybrid_indexes")

            # Rebuild FTS
            print(f"  Building FTS index...")
            try:
                new_con.execute(
                    f"PRAGMA create_fts_index('{table}', 'id', 'content', overwrite=1)"
                )
                print(f"  FTS OK")
            except Exception as e:
                print(f"  FTS WARNING: {e}")

            # Rebuild HNSW
            print(f"  Building HNSW index...")
            hnsw_idx = f"{table}_hnsw_idx"
            try:
                new_con.execute(f"DROP INDEX IF EXISTS {hnsw_idx}")
                new_con.execute(
                    f"CREATE INDEX {hnsw_idx} ON {table} USING HNSW (embedding) WITH (metric='cosine')"
                )
                print(f"  HNSW OK")
            except Exception as e:
                print(f"  HNSW WARNING: {e}")

            transferred += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1

    # ── Detach old DB ─────────────────────────────────────────────────────────
    new_con.execute("DETACH old_db")
    old_con.close()
    new_con.close()

    print(f"\n{'='*60}")
    print(f"  Transferred : {transferred}")
    print(f"  Skipped     : {skipped} (already existed)")
    print(f"  Errors      : {errors}")
    print(f"{'='*60}\n")

    if transferred > 0:
        print("Done. Restart Streamlit to use the new indexes.")


def _sanitize(name: str) -> str:
    import re
    safe = re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_")
    return safe or "default"


def _detect_dim(con, table: str) -> int:
    """Detect embedding dimension from the old table, fallback to 768."""
    try:
        row = con.execute(
            f"SELECT embedding_dim FROM old_db.{table} WHERE embedding_dim IS NOT NULL LIMIT 1"
        ).fetchone()
        if row and row[0]:
            return int(row[0])
        # Try from array length
        row = con.execute(
            f"SELECT array_length(embedding) FROM old_db.{table} WHERE embedding IS NOT NULL LIMIT 1"
        ).fetchone()
        if row and row[0]:
            return int(row[0])
    except Exception:
        pass
    return 768


if __name__ == "__main__":
    main()
