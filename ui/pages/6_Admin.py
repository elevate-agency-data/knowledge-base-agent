"""
Page 6 — Administration

Gestion des utilisateurs : créer, activer/désactiver, réinitialiser le mot de passe.
Accès réservé aux administrateurs.
"""

import path_setup  # noqa: F401
import streamlit as st
from config import APP_TITLE
from auth import require_admin, create_user, list_users, set_user_active, reset_password
from components.sidebar_auth import render_sidebar_user

st.set_page_config(
    page_title=f"Admin — {APP_TITLE}",
    page_icon=None,
    layout="wide",
)

# ── Guard admin ───────────────────────────────────────────────────────────────

require_admin()

with st.sidebar:
    st.header("Administration")
    render_sidebar_user()

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Gestion des utilisateurs")
st.divider()

# ── Créer un utilisateur ──────────────────────────────────────────────────────

st.subheader("Créer un compte")

with st.form("create_user_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        new_name  = st.text_input("Nom d'affichage")
        new_email = st.text_input("Email")
    with col2:
        new_pw      = st.text_input("Mot de passe", type="password")
        new_confirm = st.text_input("Confirmer", type="password")
    new_is_admin = st.checkbox("Compte administrateur")
    create_btn   = st.form_submit_button("Créer le compte", type="primary")

if create_btn:
    if not new_name or not new_email or not new_pw:
        st.error("Tous les champs sont requis.")
    elif new_pw != new_confirm:
        st.error("Les mots de passe ne correspondent pas.")
    elif len(new_pw) < 6:
        st.error("Mot de passe trop court (6 caractères minimum).")
    else:
        result = create_user(new_email, new_pw, new_name, is_admin=new_is_admin)
        if result:
            st.success(f"Compte créé pour **{result['email']}**.")
            st.rerun()
        else:
            st.error("Cet email est déjà utilisé.")

st.divider()

# ── Liste des utilisateurs ────────────────────────────────────────────────────

st.subheader("Utilisateurs")

users = list_users()

if not users:
    st.info("Aucun utilisateur.")
else:
    for u in users:
        col_info, col_status, col_pw, col_toggle = st.columns([4, 2, 2, 1])

        with col_info:
            badge = " 🔑" if u["is_admin"] else ""
            st.markdown(f"**{u['display_name']}**{badge}")
            st.caption(u["email"])

        with col_status:
            status = "Actif" if u["is_active"] else "Désactivé"
            if u["is_active"]:
                st.success(status)
            else:
                st.error(status)

        with col_pw:
            with st.popover("Reset mot de passe"):
                with st.form(f"pw_reset_{u['id']}"):
                    new_pw_val = st.text_input(
                        "Nouveau mot de passe", type="password",
                        key=f"pw_{u['id']}",
                    )
                    if st.form_submit_button("Confirmer"):
                        if len(new_pw_val) < 6:
                            st.error("Trop court.")
                        else:
                            reset_password(u["id"], new_pw_val)
                            st.success("Mot de passe mis à jour.")

        with col_toggle:
            # Empêche l'admin de se désactiver lui-même
            is_self = u["id"] == st.session_state.user_id
            if is_self:
                st.caption("(vous)")
            else:
                action = "Désactiver" if u["is_active"] else "Activer"
                if st.button(action, key=f"toggle_{u['id']}"):
                    set_user_active(u["id"], not u["is_active"])
                    st.rerun()

        st.divider()
