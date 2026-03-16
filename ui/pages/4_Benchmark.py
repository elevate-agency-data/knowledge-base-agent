"""
Page 4 — Benchmark Viewer

Visualise les résultats du benchmark hybrid RAG stockés dans
hybrid/data/benchmark_results.json.

Sections :
  1. Podium        — top 3 combinaisons par nDCG
  2. Leaderboard   — tableau complet trié et filtrable
  3. Par embedding — comparatif des 5 modèles (meilleur score par modèle)
  4. Par chunking  — comparatif des 3 stratégies
  5. Par retrieval — comparatif dense / sparse / hybrid
  6. Heatmap       — grille embedding × chunking colorée par nDCG
"""

from __future__ import annotations

import json
import os
import path_setup  # noqa: F401

import pandas as pd
import streamlit as st

from config import APP_TITLE, HYBRID_COLOR

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=f"Benchmark — {APP_TITLE}",
    page_icon=None,
    layout="wide",
)

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "hybrid", "data", "benchmark_results.json"
)

METRIC_LABELS = {
    "ndcg":      "nDCG@10",
    "mrr":       "MRR",
    "recall":    "Recall@10",
    "precision": "Precision@10",
}

METRIC_HELP = {
    "ndcg":      "Qualité globale du classement (1.0 = parfait). Métrique principale.",
    "mrr":       "Le bon chunk est-il en première position ?",
    "recall":    "Tous les bons chunks sont-ils retrouvés dans le top-10 ?",
    "precision": "Parmi les 10 résultats, quelle fraction est pertinente ?",
}

MODEL_LABELS = {
    "minilm-384":    "MiniLM-384",
    "mpnet-768":     "MPNet-768",
    "e5-large-1024": "E5-Large-1024",
    "bge-m3":        "BGE-M3",
    "vertex":        "Vertex (Google)",
}

CHUNK_LABELS = {
    "fixed":         "Fixed (512)",
    "fixed-128":     "Fixed 128",
    "fixed-256":     "Fixed 256",
    "fixed-512":     "Fixed 512",
    "fixed-1024":    "Fixed 1024",
    "semantic":      "Semantic",
    "hierarchical":  "Hierarchical",
}

RETRIEVAL_LABELS = {
    "dense":  "Dense",
    "sparse": "Sparse",
    "hybrid": "Hybrid (RRF)",
}

MEDAL = ["1.", "2.", "3."]


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load(path: str) -> pd.DataFrame | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            return None
        df = pd.DataFrame(data)
        # Keep only successful rows
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


def _color_score(val: float) -> str:
    if val >= 0.9:
        return "background-color: #d4edda; color: #155724"
    if val >= 0.7:
        return "background-color: #fff3cd; color: #856404"
    return "background-color: #f8d7da; color: #721c24"


def _style_df(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame.style:
    return df.style.applymap(_color_score, subset=metric_cols).format(
        {c: "{:.4f}" for c in metric_cols}
    )


# ── Load data ─────────────────────────────────────────────────────────────────

st.title("Benchmark — Résultats de retrieval")

df = _load(os.path.normpath(RESULTS_PATH))

if df is None or df.empty:
    st.warning(
        "Aucun résultat de benchmark trouvé.\n\n"
        "Lance le benchmark avec :\n"
        "```bash\n"
        "python -m hybrid.benchmark.eval_retrieval\n"
        "```\n"
        "Les résultats seront sauvegardés dans `hybrid/data/benchmark_results.json`."
    )
    st.stop()

if st.button("Recharger les résultats", type="secondary"):
    _load.clear()
    st.rerun()

st.caption(f"{len(df)} combinaisons évaluées · fichier : `hybrid/data/benchmark_results.json`")
st.divider()

# ── Sidebar — filtre métrique ─────────────────────────────────────────────────

with st.sidebar:
    st.header("Benchmark")
    st.divider()
    selected_metric = st.selectbox(
        "Métrique principale",
        list(METRIC_LABELS.keys()),
        format_func=lambda k: METRIC_LABELS[k],
        help="Métrique utilisée pour classer et colorer les graphiques.",
    )
    st.caption(METRIC_HELP[selected_metric])
    st.divider()
    st.markdown("**Filtres**")
    emb_filter = st.multiselect(
        "Modèles d'embedding",
        options=sorted(df["embedding"].unique()),
        default=sorted(df["embedding"].unique()),
        format_func=lambda k: MODEL_LABELS.get(k, k),
    )
    chunk_filter = st.multiselect(
        "Stratégies de chunking",
        options=sorted(df["chunking"].unique()),
        default=sorted(df["chunking"].unique()),
        format_func=lambda k: CHUNK_LABELS.get(k, k),
    )
    retrieval_filter = st.multiselect(
        "Modes de retrieval",
        options=sorted(df["retrieval"].unique()),
        default=sorted(df["retrieval"].unique()),
        format_func=lambda k: RETRIEVAL_LABELS.get(k, k),
    )

filtered = df[
    df["embedding"].isin(emb_filter) &
    df["chunking"].isin(chunk_filter) &
    df["retrieval"].isin(retrieval_filter)
].reset_index(drop=True)

if filtered.empty:
    st.warning("Aucun résultat avec les filtres sélectionnés.")
    st.stop()

metric_col = selected_metric
metric_label = METRIC_LABELS[selected_metric]

# ── Section 1 — Podium ────────────────────────────────────────────────────────

st.subheader(f"Podium — Top 3 par {metric_label}")
top3 = filtered.head(3)

podium_cols = st.columns(3)
for i, (_, row) in enumerate(top3.iterrows()):
    with podium_cols[i]:
        score = row[metric_col]
        st.markdown(
            f"<div style='text-align:center;padding:16px;border-radius:10px;"
            f"border:2px solid {HYBRID_COLOR};background:#f0faf4'>"
            f"<div style='font-size:2em'>{MEDAL[i]}</div>"
            f"<div style='font-weight:bold;font-size:1.1em;margin:6px 0'>{score:.4f}</div>"
            f"<div style='font-size:0.85em;color:#555'>"
            f"<b>{MODEL_LABELS.get(row['embedding'], row['embedding'])}</b><br>"
            f"{CHUNK_LABELS.get(row['chunking'], row['chunking'])} · "
            f"{RETRIEVAL_LABELS.get(row['retrieval'], row['retrieval'])}</div>"
            f"<div style='font-size:0.75em;color:#888;margin-top:6px'>"
            f"MRR {row['mrr']:.3f} · R@10 {row['recall']:.3f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── Section 2 — Leaderboard complet ──────────────────────────────────────────

st.subheader("Leaderboard complet")

display_df = filtered[["label", "ndcg", "mrr", "recall", "precision"]].copy()
display_df.index = range(1, len(display_df) + 1)
display_df.columns = ["Combinaison", "nDCG@10", "MRR", "Recall@10", "Precision@10"]

st.dataframe(
    _style_df(display_df, ["nDCG@10", "MRR", "Recall@10", "Precision@10"]),
    use_container_width=True,
    height=min(400, 45 + 35 * len(display_df)),
)

st.caption(">= 0.90 (vert)  >= 0.70 (jaune)  < 0.70 (rouge)")
st.divider()

# ── Section 3 — Comparatif par embedding ─────────────────────────────────────

st.subheader(f"Comparatif par modèle d'embedding — meilleur {metric_label}")

best_by_emb = (
    filtered.groupby("embedding")[metric_col]
    .max()
    .reset_index()
    .sort_values(metric_col, ascending=False)
)
best_by_emb["Modèle"] = best_by_emb["embedding"].map(MODEL_LABELS).fillna(best_by_emb["embedding"])
best_by_emb = best_by_emb.set_index("Modèle")[[metric_col]].rename(columns={metric_col: metric_label})

st.bar_chart(best_by_emb, color=HYBRID_COLOR, height=300)

# Tableau détaillé par embedding (toutes métriques)
with st.expander("Détail toutes métriques par embedding"):
    detail_emb = (
        filtered.groupby("embedding")[["ndcg", "mrr", "recall", "precision"]]
        .max()
        .reset_index()
        .sort_values("ndcg", ascending=False)
    )
    detail_emb["embedding"] = detail_emb["embedding"].map(MODEL_LABELS).fillna(detail_emb["embedding"])
    detail_emb.columns = ["Modèle", "nDCG@10", "MRR", "Recall@10", "Precision@10"]
    detail_emb = detail_emb.set_index("Modèle")
    st.dataframe(
        _style_df(detail_emb, ["nDCG@10", "MRR", "Recall@10", "Precision@10"]),
        use_container_width=True,
    )

st.divider()

# ── Section 4 — Comparatif par chunking ──────────────────────────────────────

st.subheader(f"Comparatif par stratégie de chunking — meilleur {metric_label}")

best_by_chunk = (
    filtered.groupby("chunking")[metric_col]
    .max()
    .reset_index()
    .sort_values(metric_col, ascending=False)
)
best_by_chunk["Chunking"] = best_by_chunk["chunking"].map(CHUNK_LABELS).fillna(best_by_chunk["chunking"])
best_by_chunk = best_by_chunk.set_index("Chunking")[[metric_col]].rename(columns={metric_col: metric_label})

st.bar_chart(best_by_chunk, color=HYBRID_COLOR, height=250)

with st.expander("Détail toutes métriques par chunking"):
    detail_chunk = (
        filtered.groupby("chunking")[["ndcg", "mrr", "recall", "precision"]]
        .max()
        .reset_index()
        .sort_values("ndcg", ascending=False)
    )
    detail_chunk["chunking"] = detail_chunk["chunking"].map(CHUNK_LABELS).fillna(detail_chunk["chunking"])
    detail_chunk.columns = ["Chunking", "nDCG@10", "MRR", "Recall@10", "Precision@10"]
    detail_chunk = detail_chunk.set_index("Chunking")
    st.dataframe(
        _style_df(detail_chunk, ["nDCG@10", "MRR", "Recall@10", "Precision@10"]),
        use_container_width=True,
    )

st.divider()

# ── Section 5 — Comparatif par retrieval ─────────────────────────────────────

st.subheader(f"Comparatif par mode de retrieval — meilleur {metric_label}")

best_by_ret = (
    filtered.groupby("retrieval")[metric_col]
    .max()
    .reset_index()
    .sort_values(metric_col, ascending=False)
)
best_by_ret["Retrieval"] = best_by_ret["retrieval"].map(RETRIEVAL_LABELS).fillna(best_by_ret["retrieval"])
best_by_ret = best_by_ret.set_index("Retrieval")[[metric_col]].rename(columns={metric_col: metric_label})

st.bar_chart(best_by_ret, color=HYBRID_COLOR, height=250)

with st.expander("Détail toutes métriques par retrieval"):
    detail_ret = (
        filtered.groupby("retrieval")[["ndcg", "mrr", "recall", "precision"]]
        .max()
        .reset_index()
        .sort_values("ndcg", ascending=False)
    )
    detail_ret["retrieval"] = detail_ret["retrieval"].map(RETRIEVAL_LABELS).fillna(detail_ret["retrieval"])
    detail_ret.columns = ["Retrieval", "nDCG@10", "MRR", "Recall@10", "Precision@10"]
    detail_ret = detail_ret.set_index("Retrieval")
    st.dataframe(
        _style_df(detail_ret, ["nDCG@10", "MRR", "Recall@10", "Precision@10"]),
        use_container_width=True,
    )

st.divider()

# ── Section 6 — Heatmap embedding × chunking ─────────────────────────────────

st.subheader(f"Heatmap — {metric_label} par embedding × chunking")
st.caption("Meilleur score parmi les 3 modes de retrieval pour chaque cellule.")

heatmap_data = (
    filtered.groupby(["embedding", "chunking"])[metric_col]
    .max()
    .unstack("chunking")
)
heatmap_data.index = heatmap_data.index.map(lambda k: MODEL_LABELS.get(k, k))
heatmap_data.columns = [CHUNK_LABELS.get(c, c) for c in heatmap_data.columns]

# Style : coloré par valeur
def _heat_color(val):
    if pd.isna(val):
        return ""
    if val >= 0.9:
        return "background-color: #28a745; color: white; font-weight: bold"
    if val >= 0.75:
        return "background-color: #85c78a; color: #155724"
    if val >= 0.6:
        return "background-color: #ffc107; color: #856404"
    return "background-color: #dc3545; color: white"

st.dataframe(
    heatmap_data.style
    .applymap(_heat_color)
    .format("{:.4f}", na_rep="—"),
    use_container_width=True,
)

st.caption(">= 0.90 (vert foncé)  >= 0.75 (vert clair)  >= 0.60 (jaune)  < 0.60 (rouge)")
st.divider()

# ── Section 7 — Recommandation ────────────────────────────────────────────────

st.subheader("Recommandation")

best = filtered.iloc[0]
st.success(
    f"**Meilleure combinaison** ({metric_label} = {best[metric_col]:.4f})  \n"
    f"**Embedding :** {MODEL_LABELS.get(best['embedding'], best['embedding'])}  \n"
    f"**Chunking :** {CHUNK_LABELS.get(best['chunking'], best['chunking'])}  \n"
    f"**Retrieval :** {RETRIEVAL_LABELS.get(best['retrieval'], best['retrieval'])}  \n\n"
    f"MRR = {best['mrr']:.4f} · Recall@10 = {best['recall']:.4f} · Precision@10 = {best['precision']:.4f}"
)

st.markdown(
    "Pour appliquer cette configuration, modifie `hybrid/config.py` :\n"
    "```python\n"
    f"DEFAULT_EMBEDDING_MODEL = \"{best['embedding']}\"\n"
    "```\n"
    f"Et utilise `chunk_strategy=\"{best['chunking']}\"` dans `hybrid_add_data` "
    f"et `retrieval_mode=\"{best['retrieval']}\"` dans `hybrid_query`."
)
