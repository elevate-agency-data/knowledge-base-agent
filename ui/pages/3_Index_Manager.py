"""
Page 3 — Index Manager

Create, inspect, populate and delete Hybrid RAG indexes.
Also allows ingesting data from Google Drive folders.
"""

from __future__ import annotations

import path_setup  # noqa: F401
import streamlit as st

from config import APP_TITLE, HYBRID_COLOR
from hybrid.config import (
    BENCHMARK_EMBEDDING_MODELS,
    BENCHMARK_CHUNK_STRATEGIES,
)

st.set_page_config(
    page_title=f"Gestion des index — {APP_TITLE}",
    page_icon=None,
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def _load_indexes():
    from services.hybrid_service import list_indexes
    return list_indexes()


def _refresh():
    _load_indexes.clear()
    st.rerun()


# ── Page header ───────────────────────────────────────────────────────────────

st.title("Gestion des index Hybrid RAG")
st.caption("Crée, inspecte, alimente et supprime tes index locaux DuckDB.")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_list, tab_create, tab_ingest, tab_inspect = st.tabs([
    "Index existants",
    "Créer un index",
    "Ingérer des données",
    "Inspecter un index",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — List existing indexes
# ═══════════════════════════════════════════════════════════════════════════════

with tab_list:
    col_title, col_refresh = st.columns([6, 1])
    with col_title:
        st.subheader("Index existants")
    with col_refresh:
        if st.button("Actualiser", help="Actualiser", use_container_width=True):
            _refresh()

    indexes = _load_indexes()

    if not indexes:
        st.info("Aucun index trouvé. Crée ton premier index dans l'onglet **Créer un index**.")
    else:
        for idx in indexes:
            name    = idx.get("index_name", "?")
            chunks  = idx.get("total_chunks", 0)
            files   = idx.get("total_files", 0)
            model   = idx.get("embedding_model", "—")
            strategy= idx.get("chunk_strategy", "—")
            created = idx.get("created_at", "")

            with st.expander(
                f"**{name}** — {chunks} chunks · {files} fichiers · `{model}`",
                expanded=False,
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Chunks",   chunks)
                c2.metric("Fichiers", files)
                c3.metric("Modèle",   model)
                c4.metric("Stratégie",strategy)
                if created:
                    st.caption(f"Créé le {str(created)[:19]}")

                # Delete button
                st.divider()
                if st.button(f"Supprimer `{name}`", key=f"del_{name}",
                             type="secondary"):
                    st.session_state[f"confirm_delete_{name}"] = True

                if st.session_state.get(f"confirm_delete_{name}"):
                    st.warning(
                        f"Confirmer la suppression de **{name}** "
                        f"({chunks} chunks) ? Cette action est irréversible."
                    )
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Confirmer", key=f"yes_{name}", type="primary"):
                            from services.hybrid_service import delete_index
                            result = delete_index(name)
                            if result.get("status") == "success":
                                st.success(result["message"])
                                st.session_state.pop(f"confirm_delete_{name}", None)
                                _refresh()
                            else:
                                st.error(result.get("message", "Erreur"))
                    with col_no:
                        if st.button("Annuler", key=f"no_{name}"):
                            st.session_state.pop(f"confirm_delete_{name}", None)
                            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Create index
# ═══════════════════════════════════════════════════════════════════════════════

with tab_create:
    st.subheader("Créer un nouvel index")

    with st.form("form_create_index"):
        new_index_name = st.text_input(
            "Nom de l'index",
            placeholder="rh",
            help="Uniquement lettres, chiffres, tirets. Sera normalisé en minuscules.",
        )
        new_model = st.selectbox(
            "Modèle d'embedding",
            BENCHMARK_EMBEDDING_MODELS,
            index=BENCHMARK_EMBEDDING_MODELS.index("mpnet-768"),
            help="mpnet-768 est recommandé pour la production. bge-m3 requiert FlagEmbedding compatible.",
        )
        new_strategy = st.selectbox(
            "Stratégie de chunking",
            BENCHMARK_CHUNK_STRATEGIES,
            help="fixed = taille fixe, semantic = frontières sémantiques, hierarchical = parent+child",
        )
        submitted = st.form_submit_button("Créer l'index", type="primary",
                                          use_container_width=True)

    if submitted:
        if not new_index_name.strip():
            st.error("Le nom de l'index est requis.")
        else:
            with st.spinner(f"Création de l'index `{new_index_name}`…"):
                from services.hybrid_service import create_index
                result = create_index(new_index_name.strip(), new_model, new_strategy)
            if result.get("status") == "success":
                st.success(result.get("message", f"Index `{new_index_name}` créé."))
                _refresh()
            else:
                st.error(result.get("message", "Erreur lors de la création."))

    st.divider()
    st.markdown("""
    **Rappels**
    - Tous les chunks d'un index doivent utiliser le même modèle d'embedding.
    - Pour changer de modèle, supprime l'index et recrée-le depuis zéro.
    - `bge-m3` requiert `pip install "transformers>=4.44.2,<5.0.0"` et `FlagEmbedding`.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Ingest data
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ingest:
    st.subheader("Ingérer des données depuis Google Drive")

    indexes = _load_indexes()
    index_names = [i["index_name"] for i in indexes]

    if not index_names:
        st.warning("Aucun index disponible. Crée un index d'abord.")
    else:
        with st.form("form_ingest"):
            target_index = st.selectbox("Index cible", index_names)

            folder_input = st.text_area(
                "Dossiers Drive (un par ligne)",
                placeholder="CELIO\nFNAC",
                help="Noms des dossiers Google Drive à indexer. L'ingestion est récursive.",
            )

            col_model, col_strategy, col_max = st.columns(3)
            with col_model:
                ingest_model = st.selectbox(
                    "Modèle d'embedding",
                    BENCHMARK_EMBEDDING_MODELS,
                    index=BENCHMARK_EMBEDDING_MODELS.index("mpnet-768"),
                    key="ingest_model",
                )
            with col_strategy:
                ingest_strategy = st.selectbox(
                    "Stratégie de chunking",
                    BENCHMARK_CHUNK_STRATEGIES,
                    key="ingest_strategy",
                )
            with col_max:
                max_files = st.number_input(
                    "Limite de fichiers (0 = illimité)",
                    min_value=0, value=0, step=10,
                    help="Utile pour tester sur un sous-ensemble.",
                )

            ingest_submitted = st.form_submit_button(
                "Lancer l'ingestion", type="primary", use_container_width=True
            )

        if ingest_submitted:
            folder_names = [f.strip() for f in folder_input.splitlines() if f.strip()]
            if not folder_names:
                st.error("Indique au moins un dossier Drive.")
            else:
                with st.spinner(
                    f"Ingestion de {len(folder_names)} dossier(s) dans `{target_index}`… "
                    "(peut prendre plusieurs minutes)"
                ):
                    from services.hybrid_service import add_data
                    result = add_data(
                        index_name=target_index,
                        folder_names=folder_names,
                        embedding_model=ingest_model,
                        chunk_strategy=ingest_strategy,
                        max_files=int(max_files),
                    )

                if result.get("status") == "success":
                    st.success(
                        f"{result.get('files_processed', '?')} fichier(s) traité(s) — "
                        f"{result.get('chunks_added', '?')} chunks ajoutés."
                    )
                    st.json(result, expanded=False)
                    _refresh()
                else:
                    st.error(result.get("message", "Erreur lors de l'ingestion."))
                    st.json(result, expanded=False)

    st.divider()

    # Drive folder browser
    st.subheader("Explorer un dossier Drive")
    drive_folder = st.text_input("Nom du dossier Drive", placeholder="Insight Factory - RAG")
    if st.button("Lister le contenu", use_container_width=False):
        if drive_folder:
            with st.spinner(f"Listage de `{drive_folder}`…"):
                from services.hybrid_service import list_drive_folder
                drive_result = list_drive_folder(drive_folder)

            if drive_result.get("status") == "success":
                folders = drive_result.get("folders", [])
                files   = drive_result.get("files", [])
                st.success(
                    f"{len(folders)} sous-dossier(s) · {len(files)} fichier(s) direct(s)"
                )
                if folders:
                    st.markdown("**Sous-dossiers**")
                    for f in folders:
                        st.caption(f"{f.get('name', '?')}")
                if files:
                    st.markdown("**Fichiers**")
                    for f in files:
                        name = f.get("name", "?")
                        url  = f.get("webViewLink", "")
                        ftype = f.get("mimeType", "").split(".")[-1]
                        if url:
                            st.caption(f"[{name}]({url})  `{ftype}`")
                        else:
                            st.caption(f"{name}  `{ftype}`")
            else:
                st.error(drive_result.get("message", "Erreur"))
        else:
            st.warning("Saisis un nom de dossier.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Inspect index
# ═══════════════════════════════════════════════════════════════════════════════

with tab_inspect:
    st.subheader("Inspecter le contenu d'un index")

    indexes = _load_indexes()
    index_names_inspect = [i["index_name"] for i in indexes]

    if not index_names_inspect:
        st.info("Aucun index disponible.")
    else:
        inspect_target = st.selectbox("Index à inspecter", index_names_inspect,
                                      key="inspect_select")

        if st.button("Charger les détails", type="primary"):
            with st.spinner(f"Chargement de `{inspect_target}`…"):
                from services.hybrid_service import get_index_info
                info = get_index_info(inspect_target)

            if info.get("status") == "success":
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Chunks",    info.get("total_chunks", 0))
                col_b.metric("Fichiers",  info.get("total_files",  0))
                col_c.metric("Modèle",    info.get("embedding_model", "—"))
                col_d.metric("Stratégie", info.get("chunk_strategy",  "—"))

                st.divider()
                files = info.get("files", [])
                if files:
                    st.markdown("**Fichiers indexés**")
                    # Build a simple table
                    import pandas as pd
                    df = pd.DataFrame(files)[
                        ["name", "type", "chunks", "langue", "domaine", "url"]
                    ]
                    df.columns = ["Fichier", "Type", "Chunks", "Langue", "Domaine", "URL"]
                    # Make URL clickable via markdown column
                    st.dataframe(
                        df,
                        use_container_width=True,
                        column_config={
                            "URL": st.column_config.LinkColumn("URL"),
                        },
                        hide_index=True,
                    )
                else:
                    st.caption("Aucun fichier trouvé.")
            else:
                st.error(info.get("message", "Erreur lors de l'inspection."))
