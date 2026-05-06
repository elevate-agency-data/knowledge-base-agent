"""
Page 3 — Knowledge Base Manager

Create, inspect, populate and delete Hybrid RAG indexes.
Displays indexes in a two-level tree: commune > category.
Also allows automatic or manual ingestion from Google Drive.
"""

from __future__ import annotations

import path_setup  # noqa: F401
import streamlit as st

from config import APP_TITLE, HYBRID_COLOR

INDEX_SEP = "__"

st.set_page_config(
    page_title=f"Knowledge Base Manager — {APP_TITLE}",
    page_icon=None,
    layout="wide",
)

from auth import require_auth
from components.sidebar_auth import render_sidebar_user
require_auth()
is_admin = st.session_state.get("is_admin", False)

with st.sidebar:
    render_sidebar_user()

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


def _category_label(index_name: str) -> str:
    """Extract the category part from 'commune__category', or return the full name."""
    if INDEX_SEP in index_name:
        return index_name.split(INDEX_SEP, 1)[1]
    return index_name


# Backwards-compat alias for older references in this file
_notion_label = _category_label


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
                    created = idx.get("created_at", "")

                    st.markdown(f"##### {notion.upper()}")
                    c1, c2 = st.columns(2)
                    c1.metric("Chunks", chunks)
                    c2.metric("Files", files)
                    if created:
                        st.caption(f"Created {str(created)[:19]}")

                    if is_admin:
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

                if is_admin:
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
        kpi1.metric("Domaines", len(grouped))
        kpi2.metric("Categories", len(all_indexes))
        kpi3.metric("Documents", grand_total_files)
        kpi4.metric("Text segments", grand_total_chunks)

        st.divider()

        # ── Theme-adaptive text color ────────────────────────────────────
        try:
            _theme = st.context.theme.base  # runtime theme (Streamlit 1.39+)
        except AttributeError:
            _theme = st.get_option("theme.base") or "light"
        _text_color = "#FFFFFF" if _theme == "dark" else "#1A1A1A"

        # ── Build sunburst data ──────────────────────────────────────────
        sb_ids: list[str] = []
        sb_labels: list[str] = []
        sb_parents: list[str] = []
        sb_values: list[int] = []
        sb_colors: list[str] = []
        sb_hover: list[str] = []

        # Distinct palette — each commune gets a clearly different color
        _PALETTE = [
            "#4285F4",  # Blue (primary)
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

        sorted_communes = sorted(grouped.items())

        for c_idx, (commune, idxs) in enumerate(sorted_communes):
            commune_files = sum(i.get("total_files", 0) for i in idxs)
            commune_chunks = sum(i.get("total_chunks", 0) for i in idxs)

            display_label = commune.upper()
            color = _PALETTE[c_idx % len(_PALETTE)]

            for idx in sorted(idxs, key=lambda x: x.get("index_name", "")):
                name = idx.get("index_name", "")
                category = _category_label(name)
                chunks = idx.get("total_chunks", 0)
                files = idx.get("total_files", 0)
                category_label = category.upper().replace("_", " ")

                # Short label: strip leading numbers, keep first word(s)
                short = category_label.lstrip("0123456789 _")
                short = short[:12].strip()
                if not short:
                    short = category_label[:10]

                sb_ids.append(name)
                sb_labels.append(short)
                sb_parents.append(commune.upper())
                sb_values.append(max(files, 1))
                sb_colors.append(color + "CC")
                sb_hover.append(
                    f"<b>{display_label} — {category_label}</b><br>"
                    f"{files} documents · {chunks} segments"
                )

            # Commune node
            sb_ids.append(commune.upper())
            sb_labels.append(display_label)
            sb_parents.append("Knowledge Base")
            sb_values.append(max(commune_files, 1))
            sb_colors.append(color)
            sb_hover.append(
                f"<b>{display_label}</b><br>"
                f"{len(idxs)} categories · {commune_files} documents<br>"
                f"{commune_chunks} text segments"
            )

        # Root node
        sb_ids.append("Knowledge Base")
        sb_labels.append("Knowledge Base")
        sb_parents.append("")
        sb_values.append(max(grand_total_files, 1))
        sb_colors.append("#EBF3FD")
        sb_hover.append(
            f"<b>Knowledge Base</b><br>"
            f"{len(grouped)} communes · {len(all_indexes)} categories<br>"
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
                color=_text_color,
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
        # Center = Knowledge Base, ring 1 = communes, ring 2 = categories
        net_nodes: list[dict] = []  # {id, label, x, y, size, color, hover}
        net_edges: list[dict] = []  # {x0, y0, x1, y1}

        # Root at center
        net_nodes.append(dict(
            id="root", label="Knowledge\nBase", x=0, y=0,
            size=40, color="#4285F4",
            hover=(
                f"<b>Knowledge Base</b><br>"
                f"{len(grouped)} communes · {len(all_indexes)} categories<br>"
                f"{grand_total_files} documents"
            ),
        ))

        n_communes = len(sorted_communes)
        commune_angle_step = 2 * math.pi / max(n_communes, 1)
        r_commune = 1.8  # radius for commune ring

        for c_idx, (commune, idxs) in enumerate(sorted_communes):
            commune_files = sum(i.get("total_files", 0) for i in idxs)
            angle = c_idx * commune_angle_step - math.pi / 2

            cx = r_commune * math.cos(angle)
            cy = r_commune * math.sin(angle)

            color = _PALETTE[c_idx % len(_PALETTE)]
            c_label = commune.upper()

            net_nodes.append(dict(
                id=commune, label=c_label, x=cx, y=cy,
                size=max(18, min(35, commune_files // 3)),
                color=color,
                hover=(
                    f"<b>{c_label}</b><br>"
                    f"{len(idxs)} categories · {commune_files} documents"
                ),
            ))
            net_edges.append(dict(x0=0, y0=0, x1=cx, y1=cy))

            # Categories around this commune
            n_categories = len(idxs)
            category_spread = min(commune_angle_step * 0.7, math.pi * 0.4)
            category_start = angle - category_spread / 2
            category_step = category_spread / max(n_categories - 1, 1) if n_categories > 1 else 0
            r_category = r_commune + 1.4

            for t_idx, idx in enumerate(sorted(idxs, key=lambda x: x.get("index_name", ""))):
                name = idx.get("index_name", "")
                category = _category_label(name)
                chunks = idx.get("total_chunks", 0)
                files = idx.get("total_files", 0)
                category_full = category.upper().replace("_", " ")
                category_short = category_full.lstrip("0123456789 ")[:10].strip()

                t_angle = category_start + t_idx * category_step if n_categories > 1 else angle
                tx = r_category * math.cos(t_angle)
                ty = r_category * math.sin(t_angle)

                # Convert hex color to rgba with transparency for categories
                _r, _g, _b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                category_color = f"rgba({_r},{_g},{_b},0.65)"

                net_nodes.append(dict(
                    id=name, label=category_short, x=tx, y=ty,
                    size=max(10, min(22, files)),
                    color=category_color,
                    hover=(
                        f"<b>{category_full}</b><br>"
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
            textfont=dict(size=10, color=_text_color),
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
            font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif", color=_text_color),
        )
        # Force text color — no selector, applies to all traces including scatter nodes
        fig_net.update_traces(textfont_color=_text_color)

        # ── Display charts ───────────────────────────────────────────────
        # CSS: force scatter text labels to follow Streamlit's theme text color
        st.markdown("""
        <style>
        .js-plotly-plot svg text {
            fill: var(--text-color) !important;
        }
        </style>
        """, unsafe_allow_html=True)

        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Network", "Sunburst", "Treemap"])
        with chart_tab1:
            st.plotly_chart(fig_net, use_container_width=True, key="network", theme=None)
        with chart_tab2:
            st.plotly_chart(fig_sun, use_container_width=True, key="sunburst")
        with chart_tab3:
            st.plotly_chart(fig_tree, use_container_width=True, key="treemap")

        st.divider()

        # ── Detail cards per commune ─────────────────────────────────────
        st.subheader("Document details")

        for commune, idxs in sorted(grouped.items()):
            commune_files = sum(i.get("total_files", 0) for i in idxs)

            with st.expander(
                f"**{commune.upper()}** — {len(idxs)} categories · "
                f"{commune_files} documents",
                expanded=False,
            ):
                sorted_idxs = sorted(idxs, key=lambda x: x.get("index_name", ""))
                for idx in sorted_idxs:
                    name = idx.get("index_name", "")
                    category = _category_label(name)
                    chunks = idx.get("total_chunks", 0)
                    files = idx.get("total_files", 0)

                    st.markdown(
                        f"<div style='border-left:3px solid #4285F4;padding:8px 12px;"
                        f"margin:8px 0;background:#EBF3FD;border-radius:0 6px 6px 0'>"
                        f"<strong style='color:#4285F4;font-size:1.05em'>"
                        f"{category.upper().replace('_', ' ')}</strong>"
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
                        st.caption("    No documents in this category yet.")

        st.caption(
            "Each commune is organized by category. "
            "Documents are automatically split into searchable text segments for AI retrieval. "
            "Click on chart segments to zoom in."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Create index
# ═══════════════════════════════════════════════════════════════════════════════

with tab_create:
    st.subheader("Create a New Index")
    if not is_admin:
        st.warning("Cette section est réservée aux administrateurs.")
    else:
        with st.form("form_create_index"):
            col_commune, col_category = st.columns(2)
            with col_commune:
                new_commune = st.text_input(
                    "Commune",
                    placeholder="paris",
                    help="Name of the commune (city hall).",
                )
            with col_category:
                new_category = st.text_input(
                    "Category",
                    placeholder="finances",
                    help="Knowledge domain: finances, rh, patrimoine, metier...",
                )

            preview_name = ""
            if new_commune.strip() and new_category.strip():
                preview_name = f"{new_commune.strip().lower()}{INDEX_SEP}{new_category.strip().lower()}"

            if preview_name:
                st.info(f"Index name: **`{preview_name}`**")

            submitted = st.form_submit_button("Create Index", type="primary",
                                              use_container_width=True)

        if submitted:
            if not new_commune.strip() or not new_category.strip():
                st.error("Both Commune and Category fields are required.")
            else:
                index_name = f"{new_commune.strip().lower()}{INDEX_SEP}{new_category.strip().lower()}"
                with st.spinner(f"Creating index `{index_name}`..."):
                    from services.hybrid_service import create_index
                    result = create_index(index_name)
                if result.get("status") == "success":
                    st.success(result.get("message", f"Index `{index_name}` created."))
                    _refresh()
                else:
                    st.error(result.get("message", "Failed to create the index."))

        st.divider()
        st.markdown("""
        **Good to know**
        - Index names follow the **`commune__category`** convention (e.g. `paris__finances`).
        - All documents in an index share the same embedding model.
        - Categories are derived from the Drive subfolder names — one subfolder = one category.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Import data
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ingest:
    st.subheader("Import Documents from Google Drive")
    if not is_admin:
        st.warning("Cette section est réservée aux administrateurs.")
    else:
        mode_auto, mode_manual = st.tabs(["Automatic Import", "Manual Import"])

        # ── Auto ingestion ────────────────────────────────────────────────────
        with mode_auto:
            st.markdown(
                "Scans your Google Drive folder structure and automatically creates one index "
                "per **commune / category** pair. Previously imported files are skipped."
            )

            with st.form("form_ingest_auto"):
                company_filter_input = st.text_input(
                    "Filter by commune (optional)",
                    placeholder="paris, lyon",
                    help="Leave empty to import everything. Separate with commas to filter.",
                )
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
                    f"commune(s): {', '.join(company_filter)}"
                    if company_filter
                    else "all communes"
                )
                with st.spinner(
                    f"Importing documents ({label})... This may take a few minutes."
                ):
                    from services.hybrid_service import add_data_auto
                    result = add_data_auto(
                        company_filter=company_filter,
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

        # ── Manual ingestion ──────────────────────────────────────────────────
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
                        placeholder="Finances\nRH",
                        help="Names of Google Drive folders to import. Subfolders are included automatically.",
                    )

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
        communes = sorted(grouped.keys())
        selected_commune = st.selectbox(
            "Commune",
            communes,
            format_func=lambda c: c.upper(),
            key="inspect_commune",
        )

        if selected_commune:
            commune_indexes = grouped[selected_commune]
            index_names_inspect = [i["index_name"] for i in commune_indexes]

            selected_index = st.selectbox(
                "Category",
                index_names_inspect,
                format_func=lambda n: _category_label(n).upper(),
                key="inspect_index",
            )

            if st.button("Load details", type="primary"):
                with st.spinner(f"Loading `{selected_index}`..."):
                    from services.hybrid_service import get_index_info
                    info = get_index_info(selected_index)

                if info.get("status") == "success":
                    col_a, col_b = st.columns(2)
                    col_a.metric("Chunks", info.get("total_chunks", 0))
                    col_b.metric("Files", info.get("total_files", 0))

                    st.divider()
                    files = info.get("files", [])
                    if files:
                        st.markdown("**Indexed files**")
                        import pandas as pd
                        df = pd.DataFrame(files)[
                            ["name", "type", "chunks", "langue", "domaine", "url"]
                        ]
                        df.columns = ["File", "Type", "Chunks", "Language", "Domain", "URL"]
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
