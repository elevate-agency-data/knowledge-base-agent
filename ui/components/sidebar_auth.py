"""
Sidebar widget — navigation, user block, and the app's one visual language.

Every protected page renders its sidebar through here, so this is where the
shared stylesheet is injected: a page cannot end up styled differently from its
neighbour by forgetting a call. Pages may append their own controls below the
navigation (a dossier list, retrieval settings…), but the chrome — paper,
navigation, user block — is identical everywhere.
"""

from __future__ import annotations

import streamlit as st


def _ensure_style() -> None:
    """Inject the shared stylesheet once per page render."""
    from components.lux_style import inject_lux_style
    inject_lux_style()


def render_sidebar_nav() -> None:
    """
    Navigation, by role (st.page_link).
    Works with position='hidden' in st.navigation().
    """
    _ensure_style()
    is_admin = st.session_state.get("is_admin", False)

    st.markdown(
        '<div class="lux-eyebrow" style="margin:2px 0 10px 0;">Navigation</div>',
        unsafe_allow_html=True,
    )

    st.page_link("pages/Home.py",             label="Accueil"       )
    st.page_link("pages/How_RAG_Works.py",    label="How RAG Works" )
    st.page_link("pages/2_RAG_Demo.py",       label="RAG Demo"      )
    st.page_link("pages/3_Index_Manager.py",  label="Knowledge Base" )
    st.page_link("pages/5_Simple_Chat.py",    label="Simple Chat"   )
    st.page_link("pages/1_Agent_Chat.py",     label="Agent Chat"    )

    if is_admin:
        st.markdown(
            '<div class="lux-eyebrow" style="margin:18px 0 8px 0;">Admin</div>',
            unsafe_allow_html=True,
        )
        st.page_link("pages/6_Admin.py",           label="Administration" )


def render_sidebar_user_info() -> None:
    """Show the signed-in user and the sign-out control (no navigation)."""
    display = st.session_state.get("display_name") or st.session_state.get("user_email", "")
    st.divider()
    st.markdown(
        f'<div class="lux-eyebrow" style="margin-bottom:6px;">Connecté</div>'
        f'<div style="font-size:.84rem;color:var(--lux-ink);margin-bottom:12px;">'
        f'{display}</div>',
        unsafe_allow_html=True,
    )
    if st.button("Se déconnecter", use_container_width=True, key="_sidebar_logout"):
        from auth import logout_user
        logout_user()


def render_sidebar_user() -> None:
    """
    Navigation + signed-in user + sign-out.
    For pages with no sidebar controls of their own (Home, Admin, …).
    Pages that add their own controls call render_sidebar_nav() at the top and
    render_sidebar_user_info() at the bottom instead.
    """
    render_sidebar_nav()
    render_sidebar_user_info()
