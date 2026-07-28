"""
Diagnose BM25 / full-text search on the active brand's store.

Answers, in order, the questions that actually explain "BM25 returns nothing":

  1. Can we open the store at all?   (DuckDB allows a single process — a running
     app holds an exclusive lock and everything else reads an empty base.)
  2. Is the `fts` extension loadable?
  3. Does every index have its FTS index built?
  4. Does a real BM25 query return rows?

Stop the Streamlit app first, then:

    set BRAND_PROFILE=hermes
    rag_venv\\Scripts\\python.exe -m scripts.diag_bm25
    rag_venv\\Scripts\\python.exe -m scripts.diag_bm25 --rebuild   (repair)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.brand import ACTIVE            # noqa: E402
from hybrid.config import DUCKDB_PATH      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose BM25 on the active store.")
    ap.add_argument("--rebuild", action="store_true",
                    help="Rebuild the FTS index of every index that lacks one.")
    ap.add_argument("--query", default="garantie",
                    help="Term to probe with (default: garantie).")
    args = ap.parse_args()

    import duckdb

    print(f"Brand : {ACTIVE.name}")
    print(f"Store : {DUCKDB_PATH}\n")

    # 1. open ---------------------------------------------------------------
    try:
        conn = duckdb.connect(DUCKDB_PATH)
    except Exception as exc:
        print("[1] OUVERTURE : ECHEC")
        print(f"    {exc}")
        print("\n    -> Une autre instance de l'app tient la base. DuckDB "
              "n'autorise qu'un seul processus.\n"
              "       Arretez l'app Streamlit puis relancez ce diagnostic.")
        return 1
    print("[1] OUVERTURE : ok")

    # 2. fts extension ------------------------------------------------------
    try:
        conn.execute("LOAD fts")
        fts_ok = True
    except Exception:
        try:
            conn.execute("INSTALL fts")
            conn.execute("LOAD fts")
            fts_ok = True
        except Exception as exc:
            fts_ok = False
            print(f"[2] EXTENSION fts : ECHEC — {exc}")
            print("    -> Sans elle, toute recherche BM25 retombe sur un "
                  "appariement de termes.")
    if fts_ok:
        print("[2] EXTENSION fts : ok")

    # 3. per-index FTS presence --------------------------------------------
    indexes = [r[0] for r in conn.execute(
        "SELECT index_name FROM hybrid_indexes ORDER BY 1").fetchall()]
    if not indexes:
        print("\n[3] Aucun index enregistre — la base est vide.")
        conn.close()
        return 1

    schemas = {r[0] for r in conn.execute(
        "SELECT schema_name FROM information_schema.schemata").fetchall()}

    print(f"\n[3] INDEX FTS ({len(indexes)} index)")
    missing = []
    for name in indexes:
        table = "chunks_" + "".join(
            c if c.isalnum() or c == "_" else "_" for c in name.lower()
        ).strip("_")
        n = conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
        has_fts = f"fts_main_{table}" in schemas
        print(f"    {'ok ' if has_fts else 'MANQUE'}  {name:22s} "
              f"{n:4d} chunks")
        if not has_fts:
            missing.append((name, table))

    if missing and args.rebuild:
        print(f"\n    Reconstruction de {len(missing)} index FTS…")
        for name, table in missing:
            try:
                conn.execute(
                    f"PRAGMA create_fts_index('{table}', 'id', 'content', "
                    f"overwrite=1)"
                )
                print(f"      reconstruit : {name}")
            except Exception as exc:
                print(f"      ECHEC {name} : {exc}")
        schemas = {r[0] for r in conn.execute(
            "SELECT schema_name FROM information_schema.schemata").fetchall()}
    elif missing:
        print(f"\n    -> {len(missing)} index sans FTS. Relancez avec --rebuild.")

    # 4. real BM25 probe ----------------------------------------------------
    print(f"\n[4] REQUETE BM25 reelle — terme « {args.query} »")
    hits = 0
    for name in indexes:
        table = "chunks_" + "".join(
            c if c.isalnum() or c == "_" else "_" for c in name.lower()
        ).strip("_")
        if f"fts_main_{table}" not in schemas:
            continue
        try:
            rows = conn.execute(
                f'SELECT c.file_name, fts_main_{table}.match_bm25(c.id, ?) AS s '
                f'FROM "{table}" c WHERE s IS NOT NULL ORDER BY s DESC LIMIT 2',
                [args.query],
            ).fetchall()
            if rows:
                hits += len(rows)
                for fn, s in rows:
                    print(f"    {name:22s} {s:6.3f}  {fn}")
        except Exception as exc:
            print(f"    {name:22s} ERREUR : {str(exc)[:80]}")

    print(f"\n    {hits} resultat(s). "
          + ("BM25 fonctionne." if hits else
             "Aucun resultat : soit le terme est absent du corpus, "
             "soit les index FTS sont a reconstruire (--rebuild)."))

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
