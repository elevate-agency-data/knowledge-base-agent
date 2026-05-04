"""
Widget sidebar — navigation contextuelle + infos utilisateur + logout.
À appeler dans le bloc `with st.sidebar:` de chaque page protégée.
"""

from __future__ import annotations

import streamlit as st


def render_sidebar_nav() -> None:
    """
    Navigation par rôle (st.page_link).
    Fonctionne avec position='hidden' dans st.navigation().
    """
    is_admin = st.session_state.get("is_admin", False)

    st.page_link("pages/Home.py",             label="Accueil"       )
    st.page_link("pages/1_Agent_Chat.py",     label="Agent Chat"    )
    st.page_link("pages/5_Simple_Chat.py",    label="Simple Chat"   )
    st.page_link("pages/3_Index_Manager.py",  label="Knowledge Base" )

    if is_admin:
        st.divider()
        st.caption("Admin")
        st.page_link("pages/6_Admin.py",           label="Administration" )


def render_sidebar_user_info() -> None:
    """Affiche uniquement le nom de l'user connecté + bouton Logout (sans nav)."""
    display = st.session_state.get("display_name") or st.session_state.get("user_email", "")
    st.divider()
    st.caption(f"Connecté : **{display}**")
    if st.button("Logout", use_container_width=True, key="_sidebar_logout"):
        from auth import logout_user
        logout_user()


def render_sidebar_user() -> None:
    """
    Affiche nav + nom de l'user connecté + bouton Logout.
    Pour les pages sans contenu sidebar custom (Home, Admin, etc.).
    Les pages avec sidebar custom (Agent Chat, Simple Chat) doivent appeler
    render_sidebar_nav() en haut et render_sidebar_user_info() en bas.
    """
    render_sidebar_nav()
    render_sidebar_user_info()
