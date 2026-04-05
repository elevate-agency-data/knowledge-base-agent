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
        st.subheader("Knowledge Base Overview")
    with col_refresh:
        if st.button("Refresh", help="Refresh", use_container_width=True, key="refresh_tree"):
            _refresh()

    grouped = _load_indexes_grouped()

    if not grouped:
        st.info("No indexes found. Import your documents in the **Import Data** tab to get started.")
    else:
        import plotly.graph_objects as go

        all_indexes = _load_indexes()
        grand_total_chunks = sum(i.get("total_chunks", 0) for i in all_indexes)
        grand_total_files = sum(i.get("total_files", 0) for i in all_indexes)

        # ── Summary KPIs ─────────────────────────────────────────────────
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Regions / Clients", len(grouped))
        kpi2.metric("Topics", len(all_indexes))
        kpi3.metric("Documents", grand_total_files)
        kpi4.metric("Text segments", grand_total_chunks)

        st.divider()

        # ── Build sunburst data ──────────────────────────────────────────
        sb_ids: list[str] = []
        sb_labels: list[str] = []
        sb_parents: list[str] = []
        sb_values: list[int] = []
        sb_colors: list[str] = []
        sb_hover: list[str] = []

        # Distinct palette — each company gets a clearly different color
        _PALETTE = [
            "#006A4E",  # Lacoste green
            "#2563EB",  # Blue
            "#D97706",  # Amber
            "#9333EA",  # Purple
            "#DC2626",  # Red
            "#0891B2",  # Cyan
            "#C026D3",  # Fuchsia
            "#059669",  # Emerald
            "#EA580C",  # Orange
            "#4F46E5",  # Indigo
        ]

        # Special label: one company is "Customer Care", the rest are "Region N"
        _CUSTOMER_CARE_KEYWORDS = {"customer care", "customer_care", "novembre", "support"}

        sorted_companies = sorted(grouped.items())
        region_num = 0

        for c_idx, (company, idxs) in enumerate(sorted_companies):
            company_files = sum(i.get("total_files", 0) for i in idxs)
            company_chunks = sum(i.get("total_chunks", 0) for i in idxs)
            color = _PALETTE[c_idx % len(_PALETTE)]

            # Determine display label
            is_cc = any(kw in company.lower() for kw in _CUSTOMER_CARE_KEYWORDS)
            if is_cc:
                display_label = "Customer Care"
            else:
                region_num += 1
                display_label = f"Region {region_num}"

            for idx in sorted(idxs, key=lambda x: x.get("index_name", "")):
                name = idx.get("index_name", "")
                notion = _notion_label(name)
                chunks = idx.get("total_chunks", 0)
                files = idx.get("total_files", 0)
                topic_label = notion.upper().replace("_", " ")

                # Short label: strip leading numbers, keep first word(s)
                short = topic_label.lstrip("0123456789 _")
                short = short[:12].strip()
                if not short:
                    short = topic_label[:10]

                sb_ids.append(name)
                sb_labels.append(short)
                sb_parents.append(company.upper())
                sb_values.append(max(files, 1))
                sb_colors.append(color + "CC")
                sb_hover.append(
                    f"<b>{display_label} — {topic_label}</b><br>"
                    f"{files} documents · {chunks} segments"
                )

            # Company node
            sb_ids.append(company.upper())
            sb_labels.append(display_label)
            sb_parents.append("Knowledge Base")
            sb_values.append(max(company_files, 1))
            sb_colors.append(color)
            sb_hover.append(
                f"<b>{display_label}</b><br>"
                f"<i>{company.upper()}</i><br>"
                f"{len(idxs)} topics · {company_files} documents<br>"
                f"{company_chunks} text segments"
            )

        # Root node
        sb_ids.append("Knowledge Base")
        sb_labels.append("Knowledge Base")
        sb_parents.append("")
        sb_values.append(max(grand_total_files, 1))
        sb_colors.append("#F5FAF7")
        sb_hover.append(
            f"<b>Knowledge Base</b><br>"
            f"{len(grouped)} regions · {len(all_indexes)} topics<br>"
            f"{grand_total_files} documents · {grand_total_chunks} segments"
        )

        # ── Sunburst chart ───────────────────────────────────────────────
        fig_sun = go.Figure(go.Sunburst(
            ids=sb_ids,
            labels=sb_labels,
            parents=sb_parents,
            values=sb_values,
            branchvalues="total",
            hovertext=sb_hover,
            hoverinfo="text",
            textinfo="label",
            insidetextorientation="horizontal",
            marker=dict(
                colors=sb_colors,
                line=dict(width=2, color="#FFFFFF"),
            ),
        ))
        fig_sun.update_layout(
            margin=dict(t=10, l=10, r=10, b=10),
            height=550,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="-apple-system, BlinkMacSystemFont, sans-serif",
                size=11,
                color="#1A1A1A",
            ),
        )

        # ── Treemap chart ────────────────────────────────────────────────
        fig_tree = go.Figure(go.Treemap(
            ids=sb_ids,
            labels=sb_labels,
            parents=sb_parents,
            values=sb_values,
            branchvalues="total",
            hovertext=sb_hover,
            hoverinfo="text",
            textinfo="label",
            texttemplate="<b>%{label}</b>",
            marker=dict(
                colors=sb_colors,
                line=dict(width=2, color="#FFFFFF"),
            ),
            tiling=dict(packing="squarify"),
        ))
        fig_tree.update_layout(
            margin=dict(t=10, l=10, r=10, b=10),
            height=550,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="-apple-system, BlinkMacSystemFont, sans-serif",
                size=12,
                color="#FFFFFF",
            ),
        )

        # ── Network graph ────────────────────────────────────────────────
        import math

        # Build node positions using a radial layout
        # Center = Knowledge Base, ring 1 = companies, ring 2 = topics
        net_nodes: list[dict] = []  # {id, label, x, y, size, color, hover}
        net_edges: list[dict] = []  # {x0, y0, x1, y1}

        # Root at center
        net_nodes.append(dict(
            id="root", label="Knowledge\nBase", x=0, y=0,
            size=40, color="#006A4E",
            hover=(
                f"<b>Knowledge Base</b><br>"
                f"{len(grouped)} regions · {len(all_indexes)} topics<br>"
                f"{grand_total_files} documents"
            ),
        ))

        n_companies = len(sorted_companies)
        company_angle_step = 2 * math.pi / max(n_companies, 1)
        r_company = 1.8  # radius for company ring

        for c_idx, (company, idxs) in enumerate(sorted_companies):
            company_files = sum(i.get("total_files", 0) for i in idxs)
            color = _PALETTE[c_idx % len(_PALETTE)]
            angle = c_idx * company_angle_step - math.pi / 2

            cx = r_company * math.cos(angle)
            cy = r_company * math.sin(angle)

            # Reuse display label from sunburst data
            is_cc = any(kw in company.lower() for kw in _CUSTOMER_CARE_KEYWORDS)
            if is_cc:
                c_label = "Customer\nCare"
            else:
                # Find region number from sb data
                c_label = next(
                    (sb_labels[i] for i, sid in enumerate(sb_ids) if sid == company.upper()),
                    company.upper()[:10],
                )
                c_label = c_label.replace(" ", "\n")

            net_nodes.append(dict(
                id=company, label=c_label, x=cx, y=cy,
                size=max(18, min(35, company_files // 3)),
                color=color,
                hover=(
                    f"<b>{c_label.replace(chr(10), ' ')}</b><br>"
                    f"<i>{company.upper()}</i><br>"
                    f"{len(idxs)} topics · {company_files} documents"
                ),
            ))
            net_edges.append(dict(x0=0, y0=0, x1=cx, y1=cy))

            # Topics around this company
            n_topics = len(idxs)
            topic_spread = min(company_angle_step * 0.7, math.pi * 0.4)
            topic_start = angle - topic_spread / 2
            topic_step = topic_spread / max(n_topics - 1, 1) if n_topics > 1 else 0
            r_topic = r_company + 1.4

            for t_idx, idx in enumerate(sorted(idxs, key=lambda x: x.get("index_name", ""))):
                name = idx.get("index_name", "")
                notion = _notion_label(name)
                chunks = idx.get("total_chunks", 0)
                files = idx.get("total_files", 0)
                topic_full = notion.upper().replace("_", " ")
                topic_short = topic_full.lstrip("0123456789 ")[:10].strip()

                t_angle = topic_start + t_idx * topic_step if n_topics > 1 else angle
                tx = r_topic * math.cos(t_angle)
                ty = r_topic * math.sin(t_angle)

                # Convert hex color to rgba with transparency for topics
                _r, _g, _b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                topic_color = f"rgba({_r},{_g},{_b},0.65)"

                net_nodes.append(dict(
                    id=name, label=topic_short, x=tx, y=ty,
                    size=max(10, min(22, files)),
                    color=topic_color,
                    hover=(
                        f"<b>{topic_full}</b><br>"
                        f"{files} documents · {chunks} segments"
                    ),
                ))
                net_edges.append(dict(x0=cx, y0=cy, x1=tx, y1=ty))

        # Build plotly figure
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        for e in net_edges:
            edge_x += [e["x0"], e["x1"], None]
            edge_y += [e["y0"], e["y1"], None]

        fig_net = go.Figure()

        # Edges
        fig_net.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=1.5, color="#C0C0C0"),
            hoverinfo="none",
        ))

        # Nodes
        fig_net.add_trace(go.Scatter(
            x=[n["x"] for n in net_nodes],
            y=[n["y"] for n in net_nodes],
            mode="markers+text",
            marker=dict(
                size=[n["size"] for n in net_nodes],
                color=[n["color"] for n in net_nodes],
                line=dict(width=2, color="#FFFFFF"),
            ),
            text=[n["label"] for n in net_nodes],
            textposition="bottom center",
            textfont=dict(size=10, color="#1A1A1A"),
            hovertext=[n["hover"] for n in net_nodes],
            hoverinfo="text",
        ))

        fig_net.update_layout(
            showlegend=False,
            margin=dict(t=10, l=10, r=10, b=10),
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False, scaleanchor="x"),
            font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif"),
        )

        # ── Display charts ───────────────────────────────────────────────
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Network", "Sunburst", "Treemap"])
        with chart_tab1:
            st.plotly_chart(fig_net, use_container_width=True, key="network")
        with chart_tab2:
            st.plotly_chart(fig_sun, use_container_width=True, key="sunburst")
        with chart_tab3:
            st.plotly_chart(fig_tree, use_container_width=True, key="treemap")

        st.divider()

        # ── Detail cards per company ─────────────────────────────────────
        st.subheader("Document details")

        for company, idxs in sorted(grouped.items()):
            company_files = sum(i.get("total_files", 0) for i in idxs)

            with st.expander(
                f"**{company.upper()}** — {len(idxs)} topics · "
                f"{company_files} documents",
                expanded=False,
            ):
                sorted_idxs = sorted(idxs, key=lambda x: x.get("index_name", ""))
                for idx in sorted_idxs:
                    name = idx.get("index_name", "")
                    notion = _notion_label(name)
                    chunks = idx.get("total_chunks", 0)
                    files = idx.get("total_files", 0)

                    st.markdown(
                        f"<div style='border-left:3px solid #006A4E;padding:8px 12px;"
                        f"margin:8px 0;background:#F5FAF7;border-radius:0 6px 6px 0'>"
                        f"<strong style='color:#006A4E;font-size:1.05em'>"
                        f"{notion.upper().replace('_', ' ')}</strong>"
                        f"<span style='color:#666;font-size:0.85em;margin-left:12px'>"
                        f"{files} documents · {chunks} segments</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    try:
                        from services.hybrid_service import get_index_info
                        info = get_index_info(name)
                        file_list = info.get("files", []) if info.get("status") == "success" else []
                    except Exception:
                        file_list = []

                    if file_list:
                        for f in file_list:
                            fname = f.get("name", "?")
                            ftype = f.get("type", "")
                            url = f.get("url", "")
                            fchunks = f.get("chunks", 0)
                            type_label = ftype.upper() if ftype else "—"
                            if url:
                                st.markdown(
                                    f"&nbsp;&nbsp;&nbsp;&nbsp;"
                                    f"[{fname}]({url})"
                                    f" &nbsp; `{type_label}` · {fchunks} segments",
                                )
                            else:
                                st.caption(
                                    f"    {fname}  ·  {type_label}  ·  {fchunks} segments"
                                )
                    else:
                        st.caption("    No documents in this topic yet.")

        st.caption(
            "Each region/client is organized by topic. "
            "Documents are automatically split into searchable text segments for AI retrieval. "
            "Click on chart segments to zoom in."
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
