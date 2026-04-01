"""
Page 3 — Index Manager

Create, inspect, populate and delete Hybrid RAG indexes.
Displays indexes in a two-level tree: company → notion.
Also allows automatic or manual ingestion from Google Drive.
"""

from __future__ import annotations

import path_setup  # noqa: F401
import streamlit as st

from config import APP_TITLE, HYBRID_COLOR
from hybrid.config import (
    BENCHMARK_EMBEDDING_MODELS,
    BENCHMARK_CHUNK_STRATEGIES,
)

INDEX_SEP = "__"

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


@st.cache_data(ttl=30, show_spinner=False)
def _load_indexes_grouped():
    from services.hybrid_service import list_indexes_grouped
    return list_indexes_grouped()


def _refresh():
    _load_indexes.clear()
    _load_indexes_grouped.clear()
    st.rerun()


def _notion_label(index_name: str) -> str:
    """Extract the notion part from 'company__notion', or return the full name."""
    if INDEX_SEP in index_name:
        return index_name.split(INDEX_SEP, 1)[1]
    return index_name


# ── Page header ───────────────────────────────────────────────────────────────

st.title("Gestion des index Hybrid RAG")
st.caption("Crée, inspecte, alimente et supprime tes index locaux DuckDB.")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_list, tab_tree, tab_create, tab_ingest, tab_inspect = st.tabs([
    "Index existants",
    "Arborescence",
    "Créer un index",
    "Ingérer des données",
    "Inspecter un index",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — List existing indexes (tree view: company → notion)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_list:
    col_title, col_refresh = st.columns([6, 1])
    with col_title:
        st.subheader("Index existants")
    with col_refresh:
        if st.button("Actualiser", help="Actualiser", use_container_width=True):
            _refresh()

    grouped = _load_indexes_grouped()

    if not grouped:
        st.info("Aucun index trouvé. Crée ton premier index dans l'onglet **Créer un index**.")
    else:
        for company, idxs in sorted(grouped.items()):
            # Aggregate totals for the company
            total_chunks = sum(i.get("total_chunks", 0) for i in idxs)
            total_files = sum(i.get("total_files", 0) for i in idxs)

            with st.expander(
                f"**{company.upper()}** — {len(idxs)} index · "
                f"{total_chunks} chunks · {total_files} fichiers",
                expanded=True,
            ):
                for idx in sorted(idxs, key=lambda x: x.get("index_name", "")):
                    name = idx.get("index_name", "?")
                    notion = _notion_label(name)
                    chunks = idx.get("total_chunks", 0)
                    files = idx.get("total_files", 0)
                    model = idx.get("embedding_model", "—")
                    strategy = idx.get("chunk_strategy", "—")
                    created = idx.get("created_at", "")

                    st.markdown(
                        f"##### {notion.upper()}"
                    )
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Chunks", chunks)
                    c2.metric("Fichiers", files)
                    c3.metric("Modèle", model)
                    c4.metric("Stratégie", strategy)
                    if created:
                        st.caption(f"Créé le {str(created)[:19]}")

                    # Delete button for this notion index
                    if st.button(
                        f"Supprimer `{name}`",
                        key=f"del_{name}",
                        type="secondary",
                    ):
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

                    st.divider()

                # Delete ALL indexes for this company
                all_names = [i["index_name"] for i in idxs]
                if len(all_names) > 1:
                    if st.button(
                        f"Supprimer tous les index de **{company.upper()}**",
                        key=f"del_company_{company}",
                        type="secondary",
                    ):
                        st.session_state[f"confirm_delete_company_{company}"] = True

                    if st.session_state.get(f"confirm_delete_company_{company}"):
                        st.warning(
                            f"Confirmer la suppression de **{len(all_names)} index** "
                            f"pour {company.upper()} ({total_chunks} chunks) ? "
                            "Cette action est irréversible."
                        )
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button("Confirmer tout", key=f"yes_company_{company}", type="primary"):
                                from services.hybrid_service import delete_index
                                errors = []
                                for n in all_names:
                                    r = delete_index(n)
                                    if r.get("status") != "success":
                                        errors.append(n)
                                if not errors:
                                    st.success(f"Tous les index de {company.upper()} supprimés.")
                                else:
                                    st.error(f"Erreur sur : {', '.join(errors)}")
                                st.session_state.pop(f"confirm_delete_company_{company}", None)
                                _refresh()
                        with col_no:
                            if st.button("Annuler", key=f"no_company_{company}"):
                                st.session_state.pop(f"confirm_delete_company_{company}", None)
                                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Tree view (visual arborescence)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_tree:
    col_title, col_refresh = st.columns([6, 1])
    with col_title:
        st.subheader("Arborescence des index")
    with col_refresh:
        if st.button("Actualiser", help="Actualiser", use_container_width=True, key="refresh_tree"):
            _refresh()

    grouped = _load_indexes_grouped()

    if not grouped:
        st.info("Aucun index trouvé.")
    else:
        # Overall stats
        all_indexes = _load_indexes()
        grand_total_chunks = sum(i.get("total_chunks", 0) for i in all_indexes)
        grand_total_files = sum(i.get("total_files", 0) for i in all_indexes)

        # CSS for the tree
        st.markdown("""
        <style>
        .tree-root {
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.8;
            padding: 16px 20px;
            background: rgba(128, 128, 128, 0.05);
            border-radius: 10px;
            border: 1px solid rgba(128, 128, 128, 0.15);
        }
        .tree-root .node-root {
            font-size: 16px;
            font-weight: 700;
        }
        .tree-root .node-company {
            font-weight: 600;
        }
        .tree-root .node-notion {
            font-weight: 400;
        }
        .tree-root .badge {
            display: inline-block;
            padding: 1px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
            font-family: -apple-system, sans-serif;
            margin-left: 6px;
        }
        .tree-root .badge-chunks {
            background: rgba(52, 168, 83, 0.15);
            color: #34A853;
        }
        .tree-root .badge-files {
            background: rgba(66, 133, 244, 0.15);
            color: #4285F4;
        }
        .tree-root .badge-indexes {
            background: rgba(251, 188, 4, 0.15);
            color: #F9AB00;
        }
        .tree-root .dim {
            opacity: 0.45;
        }
        </style>
        """, unsafe_allow_html=True)

        # Build tree HTML
        lines: list[str] = []

        # Root node
        lines.append(
            f'<span class="node-root">RAG</span>'
            f'  <span class="badge badge-indexes">{len(all_indexes)} index</span>'
            f'  <span class="badge badge-chunks">{grand_total_chunks} chunks</span>'
            f'  <span class="badge badge-files">{grand_total_files} fichiers</span>'
        )

        sorted_companies = sorted(grouped.items())
        for c_idx, (company, idxs) in enumerate(sorted_companies):
            is_last_company = c_idx == len(sorted_companies) - 1
            company_chunks = sum(i.get("total_chunks", 0) for i in idxs)
            company_files = sum(i.get("total_files", 0) for i in idxs)

            branch = "└── " if is_last_company else "├── "
            prefix = "    " if is_last_company else "│   "

            lines.append(
                f'<span class="dim">{branch}</span>'
                f'<span class="node-company">{company.upper()}</span>'
                f'  <span class="badge badge-indexes">{len(idxs)} index</span>'
                f'  <span class="badge badge-chunks">{company_chunks} chunks</span>'
                f'  <span class="badge badge-files">{company_files} fichiers</span>'
            )

            sorted_idxs = sorted(idxs, key=lambda x: x.get("index_name", ""))
            for n_idx, idx in enumerate(sorted_idxs):
                is_last_notion = n_idx == len(sorted_idxs) - 1
                notion = _notion_label(idx.get("index_name", ""))
                chunks = idx.get("total_chunks", 0)
                files = idx.get("total_files", 0)
                model = idx.get("embedding_model", "")
                strategy = idx.get("chunk_strategy", "")

                sub_branch = "└── " if is_last_notion else "├── "

                lines.append(
                    f'<span class="dim">{prefix}{sub_branch}</span>'
                    f'<span class="node-notion">{notion.upper()}</span>'
                    f'  <span class="badge badge-chunks">{chunks} chunks</span>'
                    f'  <span class="badge badge-files">{files} fichiers</span>'
                )

        tree_html = "<br>".join(lines)
        st.markdown(f'<div class="tree-root">{tree_html}</div>', unsafe_allow_html=True)

        # Legend
        st.caption(
            "Convention : `entreprise__notion`  ·  "
            "Chaque notion = un index DuckDB isolé avec son propre HNSW + FTS"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Create index
# ═══════════════════════════════════════════════════════════════════════════════

with tab_create:
    st.subheader("Créer un nouvel index")

    with st.form("form_create_index"):
        col_company, col_notion = st.columns(2)
        with col_company:
            new_company = st.text_input(
                "Entreprise",
                placeholder="celio",
                help="Nom de l'entreprise / client (niveau 1).",
            )
        with col_notion:
            new_notion = st.text_input(
                "Notion / Domaine",
                placeholder="rh",
                help="Nom du domaine (niveau 2) : rh, commercial, juridique…",
            )

        # Preview
        preview_name = ""
        if new_company.strip() and new_notion.strip():
            preview_name = f"{new_company.strip().lower()}{INDEX_SEP}{new_notion.strip().lower()}"

        if preview_name:
            st.info(f"Nom de l'index : **`{preview_name}`**")

        new_strategy = st.selectbox(
            "Stratégie de chunking",
            BENCHMARK_CHUNK_STRATEGIES,
            help="fixed = taille fixe, semantic = frontières sémantiques, hierarchical = parent+child",
        )
        submitted = st.form_submit_button("Créer l'index", type="primary",
                                          use_container_width=True)

    if submitted:
        if not new_company.strip() or not new_notion.strip():
            st.error("Les champs Entreprise et Notion sont requis.")
        else:
            index_name = f"{new_company.strip().lower()}{INDEX_SEP}{new_notion.strip().lower()}"
            with st.spinner(f"Création de l'index `{index_name}`…"):
                from services.hybrid_service import create_index
                result = create_index(index_name, "", new_strategy)
            if result.get("status") == "success":
                st.success(result.get("message", f"Index `{index_name}` créé."))
                _refresh()
            else:
                st.error(result.get("message", "Erreur lors de la création."))

    st.divider()
    st.markdown("""
    **Rappels**
    - La convention de nommage est **`entreprise__notion`** (ex: `celio__rh`).
    - Tous les chunks d'un index utilisent le même modèle d'embedding (configuré dans `config.py`).
    - Pour changer de modèle, supprime l'index et recrée-le depuis zéro.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Ingest data
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ingest:
    st.subheader("Ingérer des données depuis Google Drive")

    mode_auto, mode_manual = st.tabs(["Ingestion automatique", "Ingestion manuelle"])

    # ── Auto ingestion ────────────────────────────────────────────────────────
    with mode_auto:
        st.markdown(
            "Scanne l'arborescence Drive et crée un index **`entreprise__notion`** "
            "par couple entreprise/domaine. Les fichiers déjà indexés sont ignorés."
        )

        with st.form("form_ingest_auto"):
            company_filter_input = st.text_input(
                "Filtrer par entreprise (optionnel)",
                placeholder="celio, clientb",
                help="Laisser vide pour tout ingérer. Séparer par des virgules pour filtrer.",
            )
            col_strat, col_max = st.columns(2)
            with col_strat:
                auto_strategy = st.selectbox(
                    "Stratégie de chunking",
                    BENCHMARK_CHUNK_STRATEGIES,
                    key="auto_strategy",
                )
            with col_max:
                auto_max_files = st.number_input(
                    "Max fichiers par index (0 = illimité)",
                    min_value=0, value=0, step=10,
                    help="Utile pour les gros volumes. Rappeler l'ingestion pour continuer.",
                    key="auto_max_files",
                )

            auto_submitted = st.form_submit_button(
                "Lancer l'ingestion automatique",
                type="primary",
                use_container_width=True,
            )

        if auto_submitted:
            company_filter = [
                c.strip().lower()
                for c in company_filter_input.split(",")
                if c.strip()
            ] or None

            label = (
                f"entreprise(s) : {', '.join(company_filter)}"
                if company_filter
                else "toutes les entreprises"
            )
            with st.spinner(
                f"Ingestion automatique ({label})… Cela peut prendre plusieurs minutes."
            ):
                from services.hybrid_service import add_data_auto
                result = add_data_auto(
                    company_filter=company_filter,
                    chunk_strategy=auto_strategy,
                    max_files_per_index=int(auto_max_files),
                )

            if result.get("status") == "success":
                st.success(result.get("message", "Ingestion terminée."))

                # Show per-index results
                per_index = result.get("indexes", [])
                if per_index:
                    st.markdown("**Résultats par index :**")
                    for idx_r in per_index:
                        name = idx_r.get("index_name", "?")
                        processed = idx_r.get("files_processed", 0)
                        skipped = idx_r.get("files_already_indexed", 0)
                        updated = idx_r.get("files_updated", 0)
                        chunks = idx_r.get("chunks_created", 0)
                        status_icon = "✅" if idx_r.get("status") == "success" else "❌"
                        st.markdown(
                            f"- {status_icon} **`{name}`** — "
                            f"{processed} traité(s), {skipped} ignoré(s), "
                            f"{updated} mis à jour, {chunks} chunks"
                        )

                st.json(result, expanded=False)
                _refresh()
            else:
                st.error(result.get("message", "Erreur lors de l'ingestion."))
                st.json(result, expanded=False)

    # ── Manual ingestion ──────────────────────────────────────────────────────
    with mode_manual:
        st.markdown(
            "Ingestion manuelle : choisis un index cible et des dossiers Drive spécifiques."
        )

        indexes = _load_indexes()
        index_names = [i["index_name"] for i in indexes]

        if not index_names:
            st.warning("Aucun index disponible. Crée un index d'abord ou utilise l'ingestion automatique.")
        else:
            with st.form("form_ingest_manual"):
                target_index = st.selectbox("Index cible", index_names)

                folder_input = st.text_area(
                    "Dossiers Drive (un par ligne)",
                    placeholder="RH\nCommercial",
                    help="Noms des dossiers Google Drive à indexer. L'ingestion est récursive.",
                )

                col_strategy, col_max = st.columns(2)
                with col_strategy:
                    ingest_strategy = st.selectbox(
                        "Stratégie de chunking",
                        BENCHMARK_CHUNK_STRATEGIES,
                        key="manual_strategy",
                    )
                with col_max:
                    max_files = st.number_input(
                        "Limite de fichiers (0 = illimité)",
                        min_value=0, value=0, step=10,
                        key="manual_max_files",
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
                            chunk_strategy=ingest_strategy,
                            max_files=int(max_files),
                        )

                    if result.get("status") == "success":
                        st.success(
                            f"{result.get('files_processed', '?')} fichier(s) traité(s) — "
                            f"{result.get('chunks_created', '?')} chunks ajoutés."
                        )
                        st.json(result, expanded=False)
                        _refresh()
                    else:
                        st.error(result.get("message", "Erreur lors de l'ingestion."))
                        st.json(result, expanded=False)

    st.divider()

    # Drive folder browser
    st.subheader("Explorer un dossier Drive")
    drive_folder = st.text_input("Nom du dossier Drive", placeholder="RAG")
    if st.button("Lister le contenu", use_container_width=False):
        if drive_folder:
            with st.spinner(f"Listage de `{drive_folder}`…"):
                from services.hybrid_service import list_drive_folder
                drive_result = list_drive_folder(drive_folder)

            if drive_result.get("status") == "success":
                folders = drive_result.get("folders", [])
                files = drive_result.get("files", [])
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
                        url = f.get("webViewLink", "")
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

    grouped = _load_indexes_grouped()

    if not grouped:
        st.info("Aucun index disponible.")
    else:
        # Two-level selection: company → notion
        companies = sorted(grouped.keys())
        selected_company = st.selectbox(
            "Entreprise",
            companies,
            format_func=lambda c: c.upper(),
            key="inspect_company",
        )

        if selected_company:
            company_indexes = grouped[selected_company]
            index_names_inspect = [i["index_name"] for i in company_indexes]

            selected_index = st.selectbox(
                "Index (notion)",
                index_names_inspect,
                format_func=lambda n: _notion_label(n).upper(),
                key="inspect_index",
            )

            if st.button("Charger les détails", type="primary"):
                with st.spinner(f"Chargement de `{selected_index}`…"):
                    from services.hybrid_service import get_index_info
                    info = get_index_info(selected_index)

                if info.get("status") == "success":
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Chunks", info.get("total_chunks", 0))
                    col_b.metric("Fichiers", info.get("total_files", 0))
                    col_c.metric("Modèle", info.get("embedding_model", "—"))
                    col_d.metric("Stratégie", info.get("chunk_strategy", "—"))

                    st.divider()
                    files = info.get("files", [])
                    if files:
                        st.markdown("**Fichiers indexés**")
                        import pandas as pd
                        df = pd.DataFrame(files)[
                            ["name", "type", "chunks", "langue", "domaine", "url"]
                        ]
                        df.columns = ["Fichier", "Type", "Chunks", "Langue", "Domaine", "URL"]
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
