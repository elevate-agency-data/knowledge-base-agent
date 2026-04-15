"""
Page 0 — Login

Seuls les utilisateurs existants peuvent se connecter.
La création de comptes est réservée aux admins (page Admin).
"""

import path_setup  # noqa: F401
import streamlit as st
from config import APP_TITLE

st.set_page_config(
    page_title=f"Login — {APP_TITLE}",
    page_icon=None,
    layout="centered",
)

# Déjà connecté → accueil
if st.session_state.get("user_id"):
    st.switch_page("pages/Home.py")
    st.stop()

from auth import authenticate_user, login_user

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div style="text-align:center; padding: 32px 0 16px 0;">
        <h1 style="margin-bottom: 4px;">{APP_TITLE}</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── Formulaire de connexion ───────────────────────────────────────────────────

with st.form("login_form"):
    email     = st.text_input("Email")
    password  = st.text_input("Mot de passe", type="password")
    submitted = st.form_submit_button(
        "Se connecter", use_container_width=True, type="primary"
    )

if submitted:
    if not email or not password:
        st.error("Veuillez remplir tous les champs.")
    else:
        user = authenticate_user(email, password)
        if user:
            login_user(user)
            st.switch_page("pages/Home.py")
        else:
            st.error("Email ou mot de passe incorrect.")
