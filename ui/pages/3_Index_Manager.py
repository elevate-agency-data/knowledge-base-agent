"""
Page 3 — Knowledge Base Manager

Create, inspect, populate and delete Hybrid RAG indexes.
Displays indexes in a two-level tree: company > topic.
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
    page_title=f"Knowledge Base Manager — {APP_TITLE}",
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
    """Extract the topic part from 'company__topic', or return the full name."""
    if INDEX_SEP in index_name:
        return index_name.split(INDEX_SEP, 1)[1]
    return index_name


# ── Page header ───────────────────────────────────────────────────────────────

st.title("Knowledge Base Manager")
st.caption("Manage your document indexes — import, organize, and monitor your knowledge base.")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_list, tab_tree, tab_create, tab_ingest, tab_inspect = st.tabs([
    "Existing Indexes",
    "Tree View",
    "Create Index",
    "Import Data",
    "Inspect Index",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — List existing indexes (tree view: company > topic)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_list:
    col_title, col_refresh = st.columns([6, 1])
    with col_title:
        st.subheader("Existing Indexes")
    with col_refresh:
        if st.button("Refresh", help="Refresh", use_container_width=True):
            _refresh()

    grouped = _load_indexes_grouped()

    if not grouped:
        st.info("No indexes found yet. Create your first index in the **Create Index** tab.")
    else:
        for company, idxs in sorted(grouped.items()):
            total_chunks = sum(i.get("total_chunks", 0) for i in idxs)
            total_files = sum(i.get("total_files", 0) for i in idxs)

            with st.expander(
                f"**{company.upper()}** — {len(idxs)} indexes · "
                f"{total_chunks} chunks · {total_files} files",
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

                    st.markdown(f"##### {notion.upper()}")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Chunks", chunks)
                    c2.metric("Files", files)
                    c3.metric("Model", model)
                    c4.metric("Strategy", strategy)
                    if created:
                        st.caption(f"Created {str(created)[:19]}")

                    if st.button(f"Delete `{name}`", key=f"del_{name}", type="secondary"):
                        st.session_state[f"confirm_delete_{name}"] = True

                    if st.session_state.get(f"confirm_delete_{name}"):
                        st.warning(
                            f"Are you sure you want to delete **{name}** "
                            f"({chunks} chunks)? This cannot be undone."
                        )
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button("Confirm", key=f"yes_{name}", type="primary"):
                                from services.hybrid_service import delete_index
                                result = delete_index(name)
                                if result.get("status") == "success":
                                    st.success(result["message"])
                                    st.session_state.pop(f"confirm_delete_{name}", None)
                                    _refresh()
                                else:
                                    st.error(result.get("message", "Error"))
                        with col_no:
                            if st.button("Cancel", key=f"no_{name}"):
                                st.session_state.pop(f"confirm_delete_{name}", None)
                                st.rerun()

                    st.divider()

                all_names = [i["index_name"] for i in idxs]
                if len(all_names) > 1:
                    if st.button(
                        f"Delete all indexes for **{company.upper()}**",
                        key=f"del_company_{company}",
                        type="secondary",
                    ):
                        st.session_state[f"confirm_delete_company_{company}"] = True

                    if st.session_state.get(f"confirm_delete_company_{company}"):
                        st.warning(
                            f"Are you sure you want to delete **{len(all_names)} indexes** "
                            f"for {company.upper()} ({total_chunks} chunks)? This cannot be undone."
                        )
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button("Confirm all", key=f"yes_company_{company}", type="primary"):
                                from services.hybrid_service import delete_index
                                errors = []
                                for n in all_names:
                                    r = delete_index(n)
                                    if r.get("status") != "success":
                                        errors.append(n)
                                if not errors:
                                    st.success(f"All indexes for {company.upper()} deleted.")
                                else:
                                    st.error(f"Failed to delete: {', '.join(errors)}")
                                st.session_state.pop(f"confirm_delete_company_{company}", None)
                                _refresh()
                        with col_no:
                            if st.button("Cancel", key=f"no_company_{company}"):
                                st.session_state.pop(f"confirm_delete_company_{company}", None)
                                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Tree view
# ═══════════════════════════════════════════════════════════════════════════════

with tab_tree:
    col_title, col_refresh = st.columns([6, 1])
    with col_title:
        st.subheader("Knowledge Base Structure")
    with col_refresh:
        if st.button("Refresh", help="Refresh", use_container_width=True, key="refresh_tree"):
            _refresh()

    grouped = _load_indexes_grouped()

    if not grouped:
        st.info("No indexes found.")
    else:
        all_indexes = _load_indexes()
        grand_total_chunks = sum(i.get("total_chunks", 0) for i in all_indexes)
        grand_total_files = sum(i.get("total_files", 0) for i in all_indexes)

        st.markdown("""
        <style>
        .tree-root {
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.8;
            padding: 16px 20px;
            background: #F5FAF7;
            border-radius: 10px;
            border: 1px solid #D4E8DC;
        }
        .tree-root .node-root {
            font-size: 16px;
            font-weight: 700;
            color: #006A4E;
        }
        .tree-root .node-company {
            font-weight: 600;
            color: #006A4E;
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
            background: #E6F4ED;
            color: #006A4E;
        }
        .tree-root .badge-files {
            background: #E6F4ED;
            color: #00A651;
        }
        .tree-root .badge-indexes {
            background: #FFF8E6;
            color: #8B7335;
        }
        .tree-root .dim {
            opacity: 0.35;
            color: #006A4E;
        }
        </style>
        """, unsafe_allow_html=True)

        lines: list[str] = []

        lines.append(
            f'<span class="node-root">Knowledge Base</span>'
            f'  <span class="badge badge-indexes">{len(all_indexes)} indexes</span>'
            f'  <span class="badge badge-chunks">{grand_total_chunks} chunks</span>'
            f'  <span class="badge badge-files">{grand_total_files} files</span>'
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
                f'  <span class="badge badge-indexes">{len(idxs)} indexes</span>'
                f'  <span class="badge badge-chunks">{company_chunks} chunks</span>'
                f'  <span class="badge badge-files">{company_files} files</span>'
            )

            sorted_idxs = sorted(idxs, key=lambda x: x.get("index_name", ""))
            for n_idx, idx in enumerate(sorted_idxs):
                is_last_notion = n_idx == len(sorted_idxs) - 1
                notion = _notion_label(idx.get("index_name", ""))
                chunks = idx.get("total_chunks", 0)
                files = idx.get("total_files", 0)

                sub_branch = "└── " if is_last_notion else "├── "

                lines.append(
                    f'<span class="dim">{prefix}{sub_branch}</span>'
                    f'<span class="node-notion">{notion.upper()}</span>'
                    f'  <span class="badge badge-chunks">{chunks} chunks</span>'
                    f'  <span class="badge badge-files">{files} files</span>'
                )

        tree_html = "<br>".join(lines)
        st.markdown(f'<div class="tree-root">{tree_html}</div>', unsafe_allow_html=True)

        st.caption(
            "Each client is organized by topic. "
            "Every topic has its own dedicated search index for maximum accuracy."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Create index
# ═══════════════════════════════════════════════════════════════════════════════

with tab_create:
    st.subheader("Create a New Index")

    with st.form("form_create_index"):
        col_company, col_notion = st.columns(2)
        with col_company:
            new_company = st.text_input(
                "Client",
                placeholder="acme",
                help="Client or company name.",
            )
        with col_notion:
            new_notion = st.text_input(
                "Topic",
                placeholder="hr",
                help="Knowledge domain: hr, sales, legal, support...",
            )

        preview_name = ""
        if new_company.strip() and new_notion.strip():
            preview_name = f"{new_company.strip().lower()}{INDEX_SEP}{new_notion.strip().lower()}"

        if preview_name:
            st.info(f"Index name: **`{preview_name}`**")

        new_strategy = st.selectbox(
            "Chunking strategy",
            BENCHMARK_CHUNK_STRATEGIES,
            help="fixed = fixed size, semantic = content-aware boundaries, hierarchical = parent+child",
        )
        submitted = st.form_submit_button("Create Index", type="primary",
                                          use_container_width=True)

    if submitted:
        if not new_company.strip() or not new_notion.strip():
            st.error("Both Client and Topic fields are required.")
        else:
            index_name = f"{new_company.strip().lower()}{INDEX_SEP}{new_notion.strip().lower()}"
            with st.spinner(f"Creating index `{index_name}`..."):
                from services.hybrid_service import create_index
                result = create_index(index_name, "", new_strategy)
            if result.get("status") == "success":
                st.success(result.get("message", f"Index `{index_name}` created."))
                _refresh()
            else:
                st.error(result.get("message", "Failed to create the index."))

    st.divider()
    st.markdown("""
    **Good to know**
    - Index names follow the **`client__topic`** convention (e.g. `acme__hr`).
    - All documents in an index share the same embedding model.
    - To change the model, delete the index and recreate it.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Import data
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ingest:
    st.subheader("Import Documents from Google Drive")

    mode_auto, mode_manual = st.tabs(["Automatic Import", "Manual Import"])

    # ── Auto ingestion ────────────────────────────────────────────────────────
    with mode_auto:
        st.markdown(
            "Scans your Google Drive folder structure and automatically creates one index "
            "per **client / topic** pair. Previously imported files are skipped."
        )

        with st.form("form_ingest_auto"):
            company_filter_input = st.text_input(
                "Filter by client (optional)",
                placeholder="acme, globex",
                help="Leave empty to import everything. Separate with commas to filter.",
            )
            col_strat, col_max = st.columns(2)
            with col_strat:
                auto_strategy = st.selectbox(
                    "Chunking strategy",
                    BENCHMARK_CHUNK_STRATEGIES,
                    key="auto_strategy",
                )
            with col_max:
                auto_max_files = st.number_input(
                    "Max files per index (0 = unlimited)",
                    min_value=0, value=0, step=10,
                    help="Useful for large volumes. Re-run to continue where you left off.",
                    key="auto_max_files",
                )

            auto_submitted = st.form_submit_button(
                "Start Automatic Import",
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
                f"client(s): {', '.join(company_filter)}"
                if company_filter
                else "all clients"
            )
            with st.spinner(
                f"Importing documents ({label})... This may take a few minutes."
            ):
                from services.hybrid_service import add_data_auto
                result = add_data_auto(
                    company_filter=company_filter,
                    chunk_strategy=auto_strategy,
                    max_files_per_index=int(auto_max_files),
                )

            if result.get("status") == "success":
                st.success(result.get("message", "Import complete."))

                per_index = result.get("indexes", [])
                if per_index:
                    st.markdown("**Results by index:**")
                    for idx_r in per_index:
                        name = idx_r.get("index_name", "?")
                        processed = idx_r.get("files_processed", 0)
                        skipped = idx_r.get("files_already_indexed", 0)
                        updated = idx_r.get("files_updated", 0)
                        chunks = idx_r.get("chunks_created", 0)
                        status_icon = "✅" if idx_r.get("status") == "success" else "❌"
                        st.markdown(
                            f"- {status_icon} **`{name}`** — "
                            f"{processed} processed, {skipped} skipped, "
                            f"{updated} updated, {chunks} chunks"
                        )

                st.json(result, expanded=False)
                _refresh()
            else:
                st.error(result.get("message", "Import failed."))
                st.json(result, expanded=False)

    # ── Manual ingestion ──────────────────────────────────────────────────────
    with mode_manual:
        st.markdown(
            "Pick a target index and specify which Google Drive folders to import."
        )

        indexes = _load_indexes()
        index_names = [i["index_name"] for i in indexes]

        if not index_names:
            st.warning("No indexes available. Create one first or use the automatic import.")
        else:
            with st.form("form_ingest_manual"):
                target_index = st.selectbox("Target index", index_names)

                folder_input = st.text_area(
                    "Drive folders (one per line)",
                    placeholder="HR\nSales",
                    help="Names of Google Drive folders to import. Subfolders are included automatically.",
                )

                col_strategy, col_max = st.columns(2)
                with col_strategy:
                    ingest_strategy = st.selectbox(
                        "Chunking strategy",
                        BENCHMARK_CHUNK_STRATEGIES,
                        key="manual_strategy",
                    )
                with col_max:
                    max_files = st.number_input(
                        "File limit (0 = unlimited)",
                        min_value=0, value=0, step=10,
                        key="manual_max_files",
                    )

                ingest_submitted = st.form_submit_button(
                    "Start Import", type="primary", use_container_width=True
                )

            if ingest_submitted:
                folder_names = [f.strip() for f in folder_input.splitlines() if f.strip()]
                if not folder_names:
                    st.error("Please enter at least one Drive folder name.")
                else:
                    with st.spinner(
                        f"Importing {len(folder_names)} folder(s) into `{target_index}`... "
                        "This may take a few minutes."
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
                            f"{result.get('files_processed', '?')} file(s) processed — "
                            f"{result.get('chunks_created', '?')} chunks created."
                        )
                        st.json(result, expanded=False)
                        _refresh()
                    else:
                        st.error(result.get("message", "Import failed."))
                        st.json(result, expanded=False)

    st.divider()

    # Drive folder browser
    st.subheader("Browse Google Drive")
    drive_folder = st.text_input("Drive folder name", placeholder="RAG")
    if st.button("List contents", use_container_width=False):
        if drive_folder:
            with st.spinner(f"Listing `{drive_folder}`..."):
                from services.hybrid_service import list_drive_folder
                drive_result = list_drive_folder(drive_folder)

            if drive_result.get("status") == "success":
                folders = drive_result.get("folders", [])
                files = drive_result.get("files", [])
                st.success(f"{len(folders)} subfolder(s) · {len(files)} file(s)")
                if folders:
                    st.markdown("**Subfolders**")
                    for f in folders:
                        st.caption(f"{f.get('name', '?')}")
                if files:
                    st.markdown("**Files**")
                    for f in files:
                        name = f.get("name", "?")
                        url = f.get("webViewLink", "")
                        ftype = f.get("mimeType", "").split(".")[-1]
                        if url:
                            st.caption(f"[{name}]({url})  `{ftype}`")
                        else:
                            st.caption(f"{name}  `{ftype}`")
            else:
                st.error(drive_result.get("message", "Error"))
        else:
            st.warning("Please enter a folder name.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Inspect index
# ═══════════════════════════════════════════════════════════════════════════════

with tab_inspect:
    st.subheader("Inspect Index Contents")

    grouped = _load_indexes_grouped()

    if not grouped:
        st.info("No indexes available.")
    else:
        companies = sorted(grouped.keys())
        selected_company = st.selectbox(
            "Client",
            companies,
            format_func=lambda c: c.upper(),
            key="inspect_company",
        )

        if selected_company:
            company_indexes = grouped[selected_company]
            index_names_inspect = [i["index_name"] for i in company_indexes]

            selected_index = st.selectbox(
                "Topic",
                index_names_inspect,
                format_func=lambda n: _notion_label(n).upper(),
                key="inspect_index",
            )

            if st.button("Load details", type="primary"):
                with st.spinner(f"Loading `{selected_index}`..."):
                    from services.hybrid_service import get_index_info
                    info = get_index_info(selected_index)

                if info.get("status") == "success":
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Chunks", info.get("total_chunks", 0))
                    col_b.metric("Files", info.get("total_files", 0))
                    col_c.metric("Model", info.get("embedding_model", "—"))
                    col_d.metric("Strategy", info.get("chunk_strategy", "—"))

                    st.divider()
                    files = info.get("files", [])
                    if files:
                        st.markdown("**Indexed files**")
                        import pandas as pd
                        df = pd.DataFrame(files)[
                            ["name", "type", "chunks", "langue", "domaine", "url"]
                        ]
                        df.columns = ["File", "Type", "Chunks", "Language", "Topic", "URL"]
                        st.dataframe(
                            df,
                            use_container_width=True,
                            column_config={
                                "URL": st.column_config.LinkColumn("URL"),
                            },
                            hide_index=True,
                        )
                    else:
                        st.caption("No files found.")
                else:
                    st.error(info.get("message", "Failed to load index details."))
