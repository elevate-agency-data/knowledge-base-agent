"""
Home page — brand landing + knowledge base overview.

Layout follows luxury editorial codes (generous whitespace, display serif,
hairline rules, letter-spaced micro-labels, sparing accent). All colors come
from the active brand profile, so every brand stays visually coherent.
"""

import path_setup  # noqa: F401
import streamlit as st
from config import (
    APP_TITLE, APP_DESCRIPTION, APP_ICON, LAYOUT,
)
from auth import require_auth
from components.sidebar_auth import render_sidebar_user
from components.brand_header import brand_header
from shared.brand import ACTIVE

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
)

require_auth()

with st.sidebar:
    render_sidebar_user()

# ── Editorial styling ─────────────────────────────────────────────────────────
# Tokens and primitives live in components/lux_style.py so every page shares one
# visual language instead of re-declaring its own.

from components.lux_style import inject_lux_style

inject_lux_style()

# ── Header ────────────────────────────────────────────────────────────────────

brand_header()

st.markdown('<div class="lux-space"></div>', unsafe_allow_html=True)

# ── Manifesto ─────────────────────────────────────────────────────────────────

_HEADLINE = ACTIVE.headline or ACTIVE.subtitle

st.markdown(
    f"""
    <div class="lux-eyebrow">En quelques mots</div>
    <div class="lux-rule"></div>
    <div class="lux-h2">{_HEADLINE}</div>
    <p class="lux-lead">{APP_DESCRIPTION}</p>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="lux-space"></div>', unsafe_allow_html=True)

# ── What you can do here ──────────────────────────────────────────────────────
# One card per page of the app, written for the person doing the job rather
# than for the person who built it: what it lets you do, not how it works.

st.markdown(
    '<div class="lux-eyebrow">Ce que vous pouvez faire</div>'
    '<div class="lux-rule"></div>',
    unsafe_allow_html=True,
)

_CARDS = [
    (
        "Poser une question",
        "Décrivez la situation du client. La réponse est cherchée pour vous "
        "dans les bons documents et arrive avec ses sources, prête à relayer.",
    ),
    (
        "Montrer une pièce",
        "Joignez la photo d'un article à votre question. Elle est lue avant la "
        "recherche : ce que c'est, ce qui est abîmé, ce qui peut être conservé.",
    ),
    (
        "Échanger simplement",
        "Un échange direct, avec ou sans consultation de la base. Utile pour "
        "comparer une réponse documentée à une réponse de mémoire.",
    ),
    (
        "Comparer les recherches",
        "Voir, sur une même question, ce que remontent les différentes façons "
        "de chercher — et pourquoi les combiner donne de meilleures réponses.",
    ),
    (
        "Comprendre le principe",
        "Une visite guidée, en images, de ce qui se passe entre un document "
        "déposé et une réponse citée.",
    ),
    (
        "Alimenter la base",
        "Ajouter des documents depuis le Drive de la Maison, les ranger par "
        "domaine et suivre ce que contient la base.",
    ),
]

for row_start in (0, 3):
    for col, (title, body) in zip(
        st.columns(3, gap="large"), _CARDS[row_start:row_start + 3]
    ):
        with col:
            st.markdown(
                f"""
                <div class="lux-card">
                  <div class="lux-card-title">{title}</div>
                  <div class="lux-card-rule"></div>
                  <p class="lux-card-body">{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    if row_start == 0:
        st.markdown('<div class="lux-space-sm"></div>', unsafe_allow_html=True)

st.markdown('<div class="lux-space"></div>', unsafe_allow_html=True)

# ── Knowledge base state ──────────────────────────────────────────────────────

st.markdown(
    '<div class="lux-eyebrow">Ce que contient la base</div>'
    '<div class="lux-rule"></div>',
    unsafe_allow_html=True,
)


@st.cache_data(ttl=120, show_spinner=False)
def _get_indexes():
    from services.hybrid_service import list_indexes
    return list_indexes()


def _metric(col, value, label) -> None:
    col.markdown(
        f'<div class="lux-metric-value">{value}</div>'
        f'<div class="lux-metric-label">{label}</div>',
        unsafe_allow_html=True,
    )


# Index names are deliberately ASCII (they feed table names and substring
# matching), but that constraint must not surface as unaccented French on the
# page. Display-only restoration for the domains this base uses.
_ACCENTED = {
    "reparation": "Réparation",
    "restauration": "Restauration",
    "entretien": "Entretien",
    "garantie": "Garantie",
    "authenticite": "Authenticité",
    "personnalisation": "Personnalisation",
    "produits": "Produits",
    "commandes": "Commandes",
    "retours": "Retours",
    "procedures": "Procédures",
    "pilotage": "Pilotage",
}


def _pretty(name: str) -> str:
    """Index name → a label a reader recognises ('reparation' → 'Réparation')."""
    label = (name.split("__", 1)[1] if "__" in name else name).strip().lower()
    if label in _ACCENTED:
        return _ACCENTED[label]
    return label.replace("_", " ").capitalize()


try:
    indexes = _get_indexes()
    n_docs = sum(i.get("total_files", 0) for i in indexes)

    col_a, col_b = st.columns(2)
    _metric(col_a, len(indexes), "Domaines couverts")
    _metric(col_b, n_docs or "—", "Documents")

    st.markdown('<div class="lux-space-sm"></div>', unsafe_allow_html=True)

    if indexes:
        rows = "".join(
            f'<div class="lux-row">'
            f'<span class="lux-row-name">{_pretty(idx.get("index_name", ""))}</span>'
            f'<span class="lux-row-meta">'
            f'{idx.get("total_files") or "—"} document'
            f'{"s" if (idx.get("total_files") or 0) > 1 else ""}</span>'
            f'</div>'
            for idx in indexes
        )
        st.markdown(rows, unsafe_allow_html=True)
        st.markdown(
            '<p class="lux-lead" style="font-size:.88rem;margin-top:22px;">'
            'Chaque réponse cite les documents dont elle vient : rien n\'est '
            'inventé, tout se vérifie.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="lux-lead">La base ne contient encore aucun document. '
            'Ajoutez-en depuis la page <b>Knowledge Base</b>.</p>',
            unsafe_allow_html=True,
        )
except Exception:
    # The store is single-writer: a second running instance makes it look empty.
    st.markdown(
        '<p class="lux-lead">L\'état de la base n\'est pas consultable pour le '
        'moment. Si l\'application tourne déjà dans une autre fenêtre, fermez-la '
        'puis rechargez cette page.</p>',
        unsafe_allow_html=True,
    )
