"""
Knowledge base app — brand/identity is set by the active profile (shared/brand.py).
Point d'entrée Streamlit : router de navigation dynamique.

Toutes les pages sont toujours enregistrées (position="hidden") pour éviter
le "Page not found" quand la session expire au refresh du navigateur.
L'accès est contrôlé par les guards require_auth/require_admin dans chaque page.
La sidebar de navigation est rendue par render_sidebar_nav() (sidebar_auth.py).

Entry point: streamlit run ui/app.py
"""

import path_setup  # noqa: F401
import streamlit as st

from components.brand_header import brand_logo

# Register the brand logo globally (top-left + sidebar) — no-op if no asset.
brand_logo()


# ── Warmup — préchargement au démarrage pour éviter le cold start ─────────────

@st.cache_resource(show_spinner=False)
def _warmup():
    """
    Charge le modèle d'embedding et ouvre la connexion DuckDB en arrière-plan
    dès le lancement de l'app, avant que le premier user ne fasse une requête.
    """
    import threading

    def _load():
        try:
            from hybrid.embeddings import get_embedding_model
            from hybrid.config import DEFAULT_EMBEDDING_MODEL
            # Déclenche le chargement SentenceTransformer (~270 Mo, ~15s)
            get_embedding_model(DEFAULT_EMBEDDING_MODEL).embed_query("warmup")
        except Exception:
            pass
        try:
            from hybrid.stores import get_store
            # Ouvre la connexion DuckDB et charge les extensions VSS/FTS
            get_store()._get_conn()
        except Exception:
            pass

    threading.Thread(target=_load, daemon=True).start()


_warmup()

# Toutes les pages enregistrées — sidebar gérée manuellement par chaque page.
# Login est la page par défaut (URL racine / session expirée).
pages = [
    st.Page("pages/0_Login.py",          title="Login",          default=True),
    st.Page("pages/Home.py",             title="Home"),
    st.Page("pages/1_Agent_Chat.py",     title="Agent Chat"),
    st.Page("pages/2_RAG_Demo.py",       title="RAG Demo"),
    st.Page("pages/How_RAG_Works.py",    title="How RAG Works"),
    st.Page("pages/5_Simple_Chat.py",    title="Simple Chat"),
    st.Page("pages/3_Index_Manager.py",  title="Knowledge Base"),
    st.Page("pages/6_Admin.py",          title="Administration"),
]

pg = st.navigation(pages, position="hidden")
pg.run()
