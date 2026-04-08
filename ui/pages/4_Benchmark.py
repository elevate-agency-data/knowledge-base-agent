"""
Page 4 — Benchmark

Onglet 1 — Résultats retrieval
  Visualise les résultats du benchmark hybrid RAG stockés dans
  hybrid/data/benchmark_results.json.

Onglet 2 — Évaluation base de connaissances
  Charge un Google Sheet de questions de référence.
  Pour chaque question, les deux pipelines tournent en parallèle (comme RAG
  Comparison). Chaque résultat s'affiche dès qu'il est prêt avec un emoji
  ✅/❌ pour la réponse et la source.
  Colonnes attendues : ID, Catégorie, Question, Réponse attendue,
                       Fichier(s) source(s), Difficulté, Type
"""

from __future__ import annotations

import json
import os
import concurrent.futures
import path_setup  # noqa: F401

import pandas as pd
import streamlit as st

from config import APP_TITLE, HYBRID_COLOR, VERTEX_COLOR

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=f"Benchmark — {APP_TITLE}",
    page_icon=None,
    layout="wide",
)

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "hybrid", "data", "benchmark_results.json"
)

EVAL_HISTORY_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hybrid", "data", "eval_runs")
)
os.makedirs(EVAL_HISTORY_DIR, exist_ok=True)

METRIC_LABELS    = {"ndcg": "nDCG@10", "mrr": "MRR", "recall": "Recall@10", "precision": "Precision@10"}
METRIC_HELP      = {
    "ndcg":      "Overall ranking quality (1.0 = perfect). Primary metric.",
    "mrr":       "Is the right chunk ranked first?",
    "recall":    "Are all relevant chunks found in the top 10?",
    "precision": "What fraction of the top 10 results is relevant?",
}
MODEL_LABELS     = {"minilm-384": "MiniLM-384", "mpnet-768": "MPNet-768",
                    "e5-large-1024": "E5-Large-1024", "bge-m3": "BGE-M3", "vertex": "Naive (Google)"}
CHUNK_LABELS     = {"fixed": "Fixed (512)", "fixed-128": "Fixed 128", "fixed-256": "Fixed 256",
                    "fixed-512": "Fixed 512", "fixed-1024": "Fixed 1024",
                    "semantic": "Semantic", "hierarchical": "Hierarchical"}
RETRIEVAL_LABELS = {"dense": "Dense", "sparse": "Sparse", "hybrid": "Hybrid (RRF)"}
MEDAL            = ["1.", "2.", "3."]


# ── Shared style helpers ───────────────────────────────────────────────────────

def _color_score(val: float) -> str:
    if val >= 0.9: return "background-color: #d4edda; color: #155724"
    if val >= 0.7: return "background-color: #fff3cd; color: #856404"
    return "background-color: #f8d7da; color: #721c24"

def _style_df(df, metric_cols):
    return df.style.applymap(_color_score, subset=metric_cols).format({c: "{:.4f}" for c in metric_cols})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — benchmark results loader
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_benchmark(path: str) -> pd.DataFrame | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            return None
        df = pd.DataFrame(data)
        df = df[df.get("error", pd.Series([""] * len(df))).fillna("") == ""] if "error" in df.columns else df
        df = df[["embedding", "chunking", "retrieval", "ndcg", "mrr", "recall", "precision"]].copy()
        df = df.dropna(subset=["ndcg"])
        df["label"] = (
            df["embedding"].map(MODEL_LABELS).fillna(df["embedding"]) + " · "
            + df["chunking"].map(CHUNK_LABELS).fillna(df["chunking"]) + " · "
            + df["retrieval"].map(RETRIEVAL_LABELS).fillna(df["retrieval"])
        )
        return df.sort_values("ndcg", ascending=False).reset_index(drop=True)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — eval persistence
# ══════════════════════════════════════════════════════════════════════════════

def _save_eval_run(results: list[dict], pipeline: str, sheet_url: str, scores: dict) -> str:
    """Save an eval run to hybrid/data/eval_runs/. Returns the file path."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = {
        "timestamp":  ts,
        "pipeline":   pipeline,
        "sheet_url":  sheet_url,
        "scores":     scores,
        "results":    results,
    }
    path = os.path.join(EVAL_HISTORY_DIR, f"eval_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
    return path


def _list_eval_runs() -> list[dict]:
    """List all saved eval runs, sorted newest first."""
    runs = []
    for fname in sorted(os.listdir(EVAL_HISTORY_DIR), reverse=True):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(EVAL_HISTORY_DIR, fname)
        try:
            with open(path, encoding="utf-8") as f:
                run = json.load(f)
            runs.append({
                "file":      fname,
                "path":      path,
                "timestamp": run.get("timestamp", fname),
                "pipeline":  run.get("pipeline", "?"),
                "sheet_url": run.get("sheet_url", ""),
                "scores":    run.get("scores", {}),
                "n":         len(run.get("results", [])),
            })
        except Exception:
            pass
    return runs


def _load_eval_run(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_sheet(sheet_url: str) -> list[dict]:
    """
    Load rows from a Google Drive file (Google Sheet, Excel, or CSV).

    Strategy:
    - Extract file ID from URL
    - Fetch file metadata to detect mimeType
    - Google Sheet  → export as CSV via Drive API
    - Excel / CSV   → download binary, parse with pandas
    """
    import re, io
    import pandas as pd
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2 import service_account
    from hybrid.config import SERVICE_ACCOUNT_PATH

    # Support both /spreadsheets/d/ID and /file/d/ID and /d/ID
    m = re.search(r"/(?:spreadsheets|file)/d/([a-zA-Z0-9_-]+)", sheet_url)
    if not m:
        m = re.search(r"/d/([a-zA-Z0-9_-]+)", sheet_url)
    if not m:
        raise ValueError("Unable to extract the file ID from the URL.")
    file_id = m.group(1)

    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_PATH,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    drive = build("drive", "v3", credentials=creds)

    # Get file metadata
    meta = drive.files().get(fileId=file_id, fields="mimeType,name").execute()
    mime = meta.get("mimeType", "")

    buf = io.BytesIO()

    if mime == "application/vnd.google-apps.spreadsheet":
        # Native Google Sheet → export as CSV
        request = drive.files().export_media(fileId=file_id, mimeType="text/csv")
    elif mime in (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ):
        # Excel file → download as-is
        request = drive.files().get_media(fileId=file_id)
    elif mime == "text/csv":
        request = drive.files().get_media(fileId=file_id)
    else:
        # Try CSV export as fallback
        request = drive.files().export_media(fileId=file_id, mimeType="text/csv")

    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    buf.seek(0)

    if mime in (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ):
        df = pd.read_excel(buf, dtype=str).fillna("")
    else:
        df = pd.read_csv(buf, dtype=str).fillna("")

    df.columns = [c.strip() for c in df.columns]
    rows = df.to_dict(orient="records")
    return [r for r in rows if any(str(v).strip() for v in r.values())]


def _check_answer(expected: str, actual: str, sources: list | None = None) -> bool:
    """
    Gemini Flash YES/NO: does the model's answer contain the expected information?
    Sources are also provided so Gemini can validate cases where the expected
    answer is about finding the right document.
    """
    try:
        from vertexai.generative_models import GenerativeModel

        src_names = ""
        if sources:
            names = [
                s.get("file_name", s.get("title", s.get("uri", "")))
                for s in sources[:5]
            ]
            src_names = ", ".join(n for n in names if n)

        sources_line = f"Sources returned by the model: {src_names}\n" if src_names else ""

        r = GenerativeModel("gemini-2.0-flash-001").generate_content(
            f"Expected answer: {expected}\n"
            f"Model answer: {actual}\n"
            f"{sources_line}\n"
            "Does the model's answer (text and/or sources) contain the key factual information "
            "from the expected answer, even if the wording is different? "
            "If the expected answer refers to a document or source, also check the returned sources. "
            "Be lenient with wording: if the main fact is present, answer YES. "
            "Answer ONLY YES or NO, without explanation."
        )
        return r.text.strip().upper().startswith("YES")
    except Exception:
        return False


def _check_source(expected_path: str, sources: list, chunks: list) -> bool:
    if not expected_path.strip():
        return True
    filename = expected_path.strip().split("/")[-1]
    stem = os.path.splitext(filename)[0].lower().replace("_", " ").replace("-", " ")

    def _hit(text: str) -> bool:
        t = text.lower().replace("_", " ").replace("-", " ")
        return stem in t or filename.lower() in t

    # Check all possible key names used by both pipelines
    for s in sources:
        if any(_hit(s.get(k, "")) for k in ("title", "uri", "source_url", "file_name")):
            return True
    for c in chunks:
        if any(_hit(c.get(k, "")) for k in ("file_name", "source_url", "title", "uri")):
            return True
    return False


def _run_hybrid(question: str, top_k: int, mode: str, retrieval_query: str = "") -> dict:
    """Same logic as RAG Comparison hybrid pipeline."""
    from services.hybrid_service import query as hq, multi_query, list_indexes
    from services.vertex_service import synthesize_from_context
    from shared.index_resolver import resolve_indexes

    all_idx = [i["index_name"] for i in list_indexes()]
    resolved = resolve_indexes(question, all_idx) if all_idx else []
    if not resolved:
        return {"status": "skipped", "answer": "", "sources": [], "chunks": [], "indexes": []}

    result = (
        hq(resolved[0], question, mode, top_k, retrieval_query=retrieval_query)
        if len(resolved) == 1
        else multi_query(resolved, question, mode, top_k, retrieval_query=retrieval_query)
    )

    if result.get("status") == "success":
        context = result.get("context") or result.get("answer", "")
        if context:
            gen = synthesize_from_context(question, context)
            result["generated_answer"] = gen.get("answer", context)

    result["indexes"] = resolved
    return result


def _run_vertex(question: str, corpus_name: str, retrieval_query: str = "") -> dict:
    """Same logic as RAG Comparison vertex pipeline."""
    from services.vertex_service import query as vq
    return vq(corpus_name, question, retrieval_query=retrieval_query)


def _render_pipeline_result(
    result: dict,
    answer_ok: bool | None,
    source_ok: bool,
    pipeline: str,  # "hybrid" | "vertex"
    color: str,
) -> None:
    """Render one pipeline result with pre-computed ✅/❌ badges."""
    if result.get("status") == "skipped":
        st.caption("_Not available_")
        return
    if result.get("status") == "error":
        st.error(result.get("message", "Error"))
        return

    answer_text = result.get("generated_answer") or result.get("answer", "")
    sources     = result.get("sources", [])
    chunks      = result.get("chunks", [])
    elapsed     = result.get("elapsed_s", "?")

    a_icon = ("✅" if answer_ok else "❌") if answer_ok is not None else "—"
    s_icon = "✅" if source_ok else "❌"

    label   = "Hybrid RAG" if pipeline == "hybrid" else "Naive RAG"
    idx_str = ""
    if pipeline == "hybrid" and result.get("indexes"):
        idx_str = f" · `{'`, `'.join(result['indexes'])}`"

    st.markdown(
        f"<span style='color:{color};font-weight:bold'>{label}</span>"
        f"<span style='font-size:0.85em;color:#666'>{idx_str} · {elapsed}s</span>",
        unsafe_allow_html=True,
    )

    col_a, col_s = st.columns(2)
    with col_a:
        st.markdown(f"**Answer** {a_icon}")
    with col_s:
        st.markdown(f"**Source** {s_icon}")

    with st.expander("View answer", expanded=False):
        st.markdown(answer_text or "_No answer_")
        if sources or chunks:
            src_list = sources or chunks
            names = [s.get("file_name", s.get("title", s.get("uri", ""))) for s in src_list[:3]]
            st.caption("Sources: " + " · ".join(n for n in names if n))


# ══════════════════════════════════════════════════════════════════════════════
# Page layout
# ══════════════════════════════════════════════════════════════════════════════

st.title("Benchmark")

tab1, tab2 = st.tabs(["Retrieval Results", "Knowledge Base Evaluation"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1
# ─────────────────────────────────────────────────────────────────────────────
bench_active=False
with tab1:
    df_bench = _load_benchmark(os.path.normpath(RESULTS_PATH))

    if df_bench is None or df_bench.empty:
        st.warning(
            "No benchmark results found.\n\n"
            "Run the benchmark with:\n"
            "```bash\n"
            "python -m hybrid.benchmark.eval_retrieval\n"
            "```"
        )
    if not bench_active:
         st.warning(
            "Embedding model benchmark is currently disabled.\n\n"
        )
    else:
         if bench_active:   
            with st.sidebar:
                st.header("Benchmark")
                st.divider()
                selected_metric = st.selectbox(
                    "Primary metric", list(METRIC_LABELS.keys()),
                    format_func=lambda k: METRIC_LABELS[k],
                )
                st.caption(METRIC_HELP[selected_metric])
                st.divider()
                st.markdown("**Filtres**")
                emb_filter = st.multiselect("Embedding models",
                    options=sorted(df_bench["embedding"].unique()),
                    default=sorted(df_bench["embedding"].unique()),
                    format_func=lambda k: MODEL_LABELS.get(k, k))
                chunk_filter = st.multiselect("Chunking strategies",
                    options=sorted(df_bench["chunking"].unique()),
                    default=sorted(df_bench["chunking"].unique()),
                    format_func=lambda k: CHUNK_LABELS.get(k, k))
                retrieval_filter = st.multiselect("Modes de retrieval",
                    options=sorted(df_bench["retrieval"].unique()),
                    default=sorted(df_bench["retrieval"].unique()),
                    format_func=lambda k: RETRIEVAL_LABELS.get(k, k))

            filtered = df_bench[
                df_bench["embedding"].isin(emb_filter) &
                df_bench["chunking"].isin(chunk_filter) &
                df_bench["retrieval"].isin(retrieval_filter)
            ].reset_index(drop=True)

            if st.button("Reload", type="secondary", key="reload_t1"):
                _load_benchmark.clear(); st.rerun()

            st.caption(f"{len(df_bench)} combinations · `hybrid/data/benchmark_results.json`")
            st.divider()

            if filtered.empty:
                st.warning("No results match the selected filters.")
            else:
                mc = selected_metric
                ml = METRIC_LABELS[mc]

                st.subheader(f"Podium — Top 3 by {ml}")
                for i, (_, row) in enumerate(filtered.head(3).iterrows()):
                    with st.columns(3)[i]:
                        st.markdown(
                            f"<div style='text-align:center;padding:16px;border-radius:10px;"
                            f"border:2px solid {HYBRID_COLOR};background:#f0faf4'>"
                            f"<div style='font-size:2em'>{MEDAL[i]}</div>"
                            f"<div style='font-weight:bold;font-size:1.1em;margin:6px 0'>{row[mc]:.4f}</div>"
                            f"<div style='font-size:0.85em;color:#555'>"
                            f"<b>{MODEL_LABELS.get(row['embedding'], row['embedding'])}</b><br>"
                            f"{CHUNK_LABELS.get(row['chunking'], row['chunking'])} · "
                            f"{RETRIEVAL_LABELS.get(row['retrieval'], row['retrieval'])}</div>"
                            f"<div style='font-size:0.75em;color:#888;margin-top:6px'>"
                            f"MRR {row['mrr']:.3f} · R@10 {row['recall']:.3f}</div></div>",
                            unsafe_allow_html=True,
                        )
                st.divider()

                st.subheader("Full leaderboard")
                dd = filtered[["label","ndcg","mrr","recall","precision"]].copy()
                dd.index = range(1, len(dd)+1)
                dd.columns = ["Combination","nDCG@10","MRR","Recall@10","Precision@10"]
                st.dataframe(_style_df(dd, ["nDCG@10","MRR","Recall@10","Precision@10"]),
                            use_container_width=True, height=min(400, 45+35*len(dd)))
                st.caption(">= 0.90 (green)  >= 0.70 (yellow)  < 0.70 (red)")
                st.divider()

                for grp_col, grp_label, label_map in [
                    ("embedding", "embedding model", MODEL_LABELS),
                    ("chunking",  "chunking strategy", CHUNK_LABELS),
                    ("retrieval", "retrieval mode", RETRIEVAL_LABELS),
                ]:
                    st.subheader(f"Comparison by {grp_label} — best {ml}")
                    best = (filtered.groupby(grp_col)[mc].max().reset_index()
                            .sort_values(mc, ascending=False))
                    best["_lbl"] = best[grp_col].map(label_map).fillna(best[grp_col])
                    st.bar_chart(best.set_index("_lbl")[[mc]].rename(columns={mc: ml}),
                                color=HYBRID_COLOR, height=260)
                    st.divider()

                st.subheader(f"Heatmap — {ml} by embedding x chunking")
                hm = (filtered.groupby(["embedding","chunking"])[mc].max().unstack("chunking"))
                hm.index   = hm.index.map(lambda k: MODEL_LABELS.get(k, k))
                hm.columns = [CHUNK_LABELS.get(c, c) for c in hm.columns]
                def _hc(v):
                    if pd.isna(v): return ""
                    if v >= 0.9:  return "background-color:#28a745;color:white;font-weight:bold"
                    if v >= 0.75: return "background-color:#85c78a;color:#155724"
                    if v >= 0.6:  return "background-color:#ffc107;color:#856404"
                    return "background-color:#dc3545;color:white"
                st.dataframe(hm.style.applymap(_hc).format("{:.4f}", na_rep="—"), use_container_width=True)
                st.caption(">= 0.90 (dark green)  >= 0.75 (light green)  >= 0.60 (yellow)  < 0.60 (red)")
                st.divider()

                st.subheader("Recommendation")
                best_row = filtered.iloc[0]
                st.success(
                    f"**Best combination** ({ml} = {best_row[mc]:.4f})  \n"
                    f"**Embedding:** {MODEL_LABELS.get(best_row['embedding'], best_row['embedding'])}  \n"
                    f"**Chunking:** {CHUNK_LABELS.get(best_row['chunking'], best_row['chunking'])}  \n"
                    f"**Retrieval:** {RETRIEVAL_LABELS.get(best_row['retrieval'], best_row['retrieval'])}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — live evaluation
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Knowledge Base Evaluation")
    st.caption(
        "For each question in the Google Sheet, both pipelines run in parallel. "
        "Results are displayed as they complete, with a pass/fail indicator for the answer and the source."
    )

    # ── Config inputs ─────────────────────────────────────────────────────────
    sheet_url = st.text_input(
        "URL Google Sheet",
        placeholder="https://docs.google.com/spreadsheets/d/...",
        help="Columns: ID, Category, Question, Expected answer, Source file(s), Difficulty, Type",
        key="eval_sheet_url",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        eval_top_k = st.slider("Top-K chunks (Hybrid)", 5, 20, 10, key="eval_topk")
    with c2:
        eval_mode = st.selectbox("Mode retrieval (Hybrid)", ["hybrid","dense","sparse"], key="eval_mode")
    with c3:
        try:
            from services.vertex_service import list_corpora as _lc
            _corpora = _lc()
            _copts   = [c["display_name"] for c in _corpora]
        except Exception:
            _copts = []
        if _copts:
            eval_corpus = st.selectbox("Corpus Naive RAG", _copts, key="eval_corpus")
        else:
            eval_corpus = st.text_input("Corpus Naive RAG (nom)", key="eval_corpus_txt", placeholder="base-rag")

    eval_pipeline = "Both"

    run_btn = st.button("Run evaluation", type="primary",
                        disabled=not sheet_url, key="eval_run")

    if run_btn and sheet_url:
        # ── Load questions ────────────────────────────────────────────────────
        with st.spinner("Loading Google Sheet..."):
            try:
                questions = _load_sheet(sheet_url)
            except Exception as exc:
                st.error(f"Unable to read the sheet: {exc}")
                questions = []

        if not questions:
            st.error("No questions found. Check the URL and permissions.")
        else:
            st.caption(f"{len(questions)} questions — starting...")
            st.divider()

            # Init vertex once
            try:
                from services.vertex_service import _init_vertex; _init_vertex()
            except Exception:
                pass

            # Accumulators
            scores     = {"H_answer": [], "H_source": [], "V_answer": [], "V_source": []}
            all_rows   = []   # collected for JSON save
            score_ph   = st.empty()

            def _pct(lst): return f"{round(sum(lst)/len(lst)*100,1)}%" if lst else "—"

            # ── Column headers ─────────────────────────────────────────────
            hdr_h, hdr_v = st.columns(2)
            with hdr_h:
                st.markdown(f"<h4 style='color:{HYBRID_COLOR}'>Hybrid RAG</h4>", unsafe_allow_html=True)
            with hdr_v:
                st.markdown(f"<h4 style='color:{VERTEX_COLOR}'>Naive RAG</h4>", unsafe_allow_html=True)
            st.divider()

            # ── Process each question ─────────────────────────────────────
            for idx, q in enumerate(questions):
                q_text  = q.get("Question", "").strip()
                q_id    = q.get("ID", f"Q{idx+1}")
                exp_ans = q.get("Réponse attendue", "").strip()
                exp_src = q.get("Fichier(s) source(s)", "").strip()
                cat     = q.get("Catégorie", "")
                diff    = q.get("Difficulté", "")

                st.markdown(f"**{q_id}** · `{cat}` · `{diff}`  \n{q_text}")

                col_h, col_v = st.columns(2)
                with col_h:
                    h_ph = st.empty(); h_ph.info("Hybrid in progress...")
                with col_v:
                    v_ph = st.empty(); v_ph.info("Naive RAG in progress...")

                h_ao = h_so = v_ao = v_so = None
                h_ans_txt = v_ans_txt = ""
                h_srcs = v_srcs = ""
                h_elapsed = v_elapsed = "?"

                # Single shared rewrite — same query for both pipelines
                try:
                    from shared.query_rewriter import rewrite_query
                    retrieval_q = rewrite_query(q_text)
                except Exception:
                    retrieval_q = q_text

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    futs = {
                        ex.submit(_run_hybrid, q_text, eval_top_k, eval_mode, retrieval_q): "hybrid",
                        ex.submit(_run_vertex, q_text, eval_corpus, retrieval_q): "vertex",
                    }
                    for fut in concurrent.futures.as_completed(futs):
                        which = futs[fut]
                        try:
                            res = fut.result()
                        except Exception as exc:
                            res = {"status": "error", "message": str(exc)}

                        ans_text = res.get("generated_answer") or res.get("answer", "")
                        all_srcs = res.get("sources", []) + res.get("chunks", [])
                        ao = _check_answer(exp_ans, ans_text, all_srcs) if (exp_ans and ans_text) else None
                        so = _check_source(exp_src, res.get("sources",[]), res.get("chunks",[]))
                        src_names = ", ".join(
                            s.get("file_name", s.get("title", s.get("uri", "")))
                            for s in (res.get("sources") or res.get("chunks") or [])[:3]
                        )

                        if which == "hybrid":
                            h_ao, h_so = ao, so
                            h_ans_txt  = ans_text
                            h_srcs     = src_names
                            h_elapsed  = res.get("elapsed_s", "?")
                            with h_ph.container():
                                _render_pipeline_result(res, ao, so, "hybrid", HYBRID_COLOR)
                        else:
                            v_ao, v_so = ao, so
                            v_ans_txt  = ans_text
                            v_srcs     = src_names
                            v_elapsed  = res.get("elapsed_s", "?")
                            with v_ph.container():
                                _render_pipeline_result(res, ao, so, "vertex", VERTEX_COLOR)

                # Accumulate scores (only when checks were actually computed)
                if h_ao is not None: scores["H_answer"].append(h_ao)
                if v_ao is not None: scores["V_answer"].append(v_ao)
                if h_so is not None: scores["H_source"].append(bool(h_so))
                if v_so is not None: scores["V_source"].append(bool(v_so))

                # Collect full row for save
                all_rows.append({
                    "id": q_id, "category": cat, "difficulty": diff,
                    "question": q_text, "expected_answer": exp_ans, "expected_source": exp_src,
                    "h_answer": h_ans_txt, "h_answer_ok": h_ao, "h_source_ok": h_so,
                    "h_sources": h_srcs,   "h_elapsed": h_elapsed,
                    "v_answer": v_ans_txt, "v_answer_ok": v_ao, "v_source_ok": v_so,
                    "v_sources": v_srcs,   "v_elapsed": v_elapsed,
                })

                score_ph.markdown(
                    f"**Score in progress** ({idx+1}/{len(questions)}) — "
                    f"<span style='color:{HYBRID_COLOR}'>Hybrid</span> "
                    f"Answer {_pct(scores['H_answer'])} · Source {_pct(scores['H_source'])} &nbsp;|&nbsp; "
                    f"<span style='color:{VERTEX_COLOR}'>Naive</span> "
                    f"Answer {_pct(scores['V_answer'])} · Source {_pct(scores['V_source'])}",
                    unsafe_allow_html=True,
                )
                st.divider()

            # ── Final score ───────────────────────────────────────────────
            score_ph.empty()
            final_scores = {k: _pct(v) for k, v in scores.items()}
            st.subheader("Final results")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Hybrid — Answer", final_scores["H_answer"])
            m2.metric("Hybrid — Source",  final_scores["H_source"])
            m3.metric("Naive — Answer", final_scores["V_answer"])
            m4.metric("Naive — Source",  final_scores["V_source"])

            # ── Save run ──────────────────────────────────────────────────
            saved_path = _save_eval_run(all_rows, eval_pipeline, sheet_url, final_scores)
            st.success(f"Results saved to `{os.path.basename(saved_path)}`")

    # ── Historique des runs ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Evaluation history")

    runs = _list_eval_runs()
    if not runs:
        st.caption("No saved runs.")
    else:
        for run in runs:
            sc = run["scores"]
            label = (
                f"`{run['timestamp']}` · {run['pipeline']} · {run['n']} questions  "
                f"— H Ans {sc.get('H_answer','—')} · H Src {sc.get('H_source','—')} "
                f"| V Ans {sc.get('V_answer','—')} · V Src {sc.get('V_source','—')}"
            )
            if st.button(f"Load {run['timestamp']}", key=f"load_{run['file']}"):
                loaded = _load_eval_run(run["path"])
                st.session_state["loaded_eval"] = loaded
                st.rerun()

            st.caption(label)

    # ── Affichage d'un run chargé ──────────────────────────────────────────────
    if "loaded_eval" in st.session_state and not run_btn:
        run = st.session_state["loaded_eval"]
        st.divider()
        st.subheader(f"Loaded run — {run.get('timestamp','')} · {run.get('pipeline','')}")
        sc = run.get("scores", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hybrid — Answer", sc.get("H_answer", "—"))
        m2.metric("Hybrid — Source",  sc.get("H_source", "—"))
        m3.metric("Naive — Answer", sc.get("V_answer", "—"))
        m4.metric("Naive — Source",  sc.get("V_source", "—"))

        def _ico(v): return ("✅" if v else "❌") if v is not None else "—"
        rows = run.get("results", [])
        if rows:
            st.divider()
            for row in rows:
                with st.expander(
                    f"{row.get('id','')} · {row.get('category','')} · {row.get('difficulty','')} "
                    f"— H {_ico(row.get('h_answer_ok'))} {_ico(row.get('h_source_ok'))} "
                    f"| V {_ico(row.get('v_answer_ok'))} {_ico(row.get('v_source_ok'))}  "
                    f"— {row.get('question','')[:70]}",
                    expanded=False,
                ):
                    st.markdown(f"**Expected:** {row.get('expected_answer','')}")
                    st.markdown(f"**Expected source:** `{row.get('expected_source','')}`")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(
                            f"<span style='color:{HYBRID_COLOR}'>**Hybrid**</span> "
                            f"Ans {_ico(row.get('h_answer_ok'))} · Src {_ico(row.get('h_source_ok'))} · {row.get('h_elapsed','?')}s",
                            unsafe_allow_html=True,
                        )
                        st.caption(row.get("h_answer","")[:300])
                        st.caption(f"Sources: {row.get('h_sources','—')}")
                    with c2:
                        st.markdown(
                            f"<span style='color:{VERTEX_COLOR}'>**Vertex**</span> "
                            f"Ans {_ico(row.get('v_answer_ok'))} · Src {_ico(row.get('v_source_ok'))} · {row.get('v_elapsed','?')}s",
                            unsafe_allow_html=True,
                        )
                        st.caption(row.get("v_answer","")[:300])
                        st.caption(f"Sources: {row.get('v_sources','—')}")
