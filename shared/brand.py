"""
Brand — single source of truth for product identity, theme AND domain prompts.

Rebrand / re-domain the whole app by editing THIS file only. Every UI label,
theme color, agent name, Drive root folder AND the domain-specific parts of the
RAG prompts (audience, business domains, retrieval vocabulary) are derived from
the *active profile* selected below.

Switching identity
------------------
Set the env var ``BRAND_PROFILE`` (``hermes`` | ``indica``) — or change
``_DEFAULT_PROFILE`` — to swap the whole identity: name, theme, and every
domain-specific prompt fragment. The generic RAG machinery (citation format,
output rules, retrieval mechanics) lives in the consumer modules and never
changes between profiles.

Adding a profile
----------------
Copy one ``BrandProfile(...)`` block, translate the identity + domain fields to
the new business case, register it in ``PROFILES``. Nothing else to touch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# ── Theme ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Theme:
    """UI palette for a profile.

    ``accent`` drives the Streamlit *primaryColor* (buttons, widgets, links)
    and the custom components. The ``background`` / ``text`` / ``base`` fields
    feed Streamlit's native theme so the whole chrome follows the brand — see
    ``streamlit_theme_env``.
    """
    accent: str          # primary accent → Streamlit primaryColor
    accent_light: str    # light accent tint (secondary background default)
    hybrid: str          # Hybrid pipeline accent
    user: str            # user message bg
    tool: str            # tool call bg
    error: str           # error bg
    background: str = "#FFFFFF"        # page background
    secondary_background: str = "#F5F6F8"  # sidebar / cards (neutral grey)
    text: str = "#1A1A1A"              # main text color
    base: str = "light"                # Streamlit base theme ("light" | "dark")


def streamlit_theme_env(theme: Theme) -> dict[str, str]:
    """Map a Theme to Streamlit ``STREAMLIT_THEME_*`` environment variables.

    Setting these before ``streamlit run`` themes the whole native chrome
    (buttons, widgets, sidebar, primary accents) per brand — overriding any
    server-level ``.streamlit/config.toml``.
    """
    return {
        "STREAMLIT_THEME_BASE": theme.base,
        "STREAMLIT_THEME_PRIMARY_COLOR": theme.accent,
        "STREAMLIT_THEME_BACKGROUND_COLOR": theme.background,
        "STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR": (
            theme.secondary_background or theme.accent_light
        ),
        "STREAMLIT_THEME_TEXT_COLOR": theme.text,
    }


# ── Profile ───────────────────────────────────────────────────────────────────
# A profile carries three concerns:
#   1. Identity   — what the product is called / where its data lives.
#   2. Theme      — the visual palette.
#   3. Domain     — the business-specific prompt fragments injected into the
#                   otherwise-generic RAG prompts (agent instruction, answer
#                   generation, query rewriter, index resolver).
#
# The BOUNDARY_RULES / PILOTAGE type aliases:
#   pilotage_labels : index-label keywords for cross-cutting / dashboard indexes
#                     that must always be added on top of the resolver result.
#   boundary_rules  : list of (trigger_terms, domain_label_keywords) — when any
#                     trigger appears in the query, indexes matching the domain
#                     label keywords are force-added (topics spanning 2 domains).
@dataclass(frozen=True)
class BrandProfile:
    # 1. Identity
    name: str
    domain: str                 # human-readable business domain
    subtitle: str               # tagline under the title
    description: str            # longer pitch (Home page)
    drive_root_folder: str      # MUST match the real Google Drive folder
    duckdb_path: str            # per-tenant DuckDB store (data isolation)

    # 2. Theme
    theme: Theme

    # 3. Domain prompt fragments
    audience: str               # who the users are
    scope: str                  # what the knowledge base covers (one clause)
    domain_examples: str        # example data domains (free text)
    rewriter_intro: str         # 1-2 sentences framing the rewriter's corpus
    rewriter_keep_terms: str    # comma list of domain terms to always keep
    rewriter_examples: str      # few-shot rewrite examples block
    resolver_domain_map: str    # bulleted 'keyword → domain' routing block
    pilotage_labels: tuple[str, ...] = ()
    boundary_rules: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = ()

    # 4. Auth & RBAC (per-brand isolation)
    #   users_db_path : separate SQLite user store per brand (accounts isolated)
    #   roles         : business roles for this brand (ordered; [0] = full owner)
    #   default_role  : role assigned to a new / unknown user (fallback)
    #   role_keywords : role → index-category keywords (None = full access)
    users_db_path: str = "ui/data/users.db"
    roles: tuple[str, ...] = ()
    default_role: str = "agent"
    role_keywords: dict[str, tuple[str, ...] | None] = field(default_factory=dict)

    @property
    def agent_name(self) -> str:
        # ADK requires a valid Python identifier — strip spaces / punctuation.
        ident = "".join(ch for ch in self.name if ch.isalnum()) or "KB"
        if ident[0].isdigit():
            ident = f"_{ident}"
        return f"{ident}Agent"


# ── Profile: Hermes (luxury Maison — after-sales / SAV & client care) ──────────
# Same customer-care model as Lacoste, adapted to a luxury house: repair &
# restoration workshop, lifetime care of leather / silk / metal, savoir-faire,
# authenticity & traceability, personalization. High-touch, no-discount register.
_HERMES = BrandProfile(
    name="Hermes",
    domain="luxury after-sales & client care (SAV)",
    subtitle="After-sales & client care knowledge base",
    description=(
        "Hermes indexes the Maison's after-sales & client care knowledge base "
        "— repair & restoration, lifetime care of leather, silk and precious "
        "metals, warranty, authenticity, personalization, orders and client "
        "requests — and lets client advisors and workshop artisans find "
        "refined, sourced answers."
    ),
    drive_root_folder="Rag_hermes",
    duckdb_path="hybrid/data/hermes_hybrid.duckdb",
    theme=Theme(
        accent="#F37021",       # Hermès orange
        accent_light="#FDEEE1",  # light orange tint
        hybrid="#8A6D3B",       # warm leather brown
        user="#FBF7F2",         # cream
        tool="#FDEEE1",
        error="#F8D7DA",
    ),
    audience=(
        "client advisors, after-sales specialists, boutique staff and "
        "workshop artisans (artisans / craftsmen)"
    ),
    scope="the after-sales (SAV) and client care service of a luxury Maison",
    domain_examples=(
        "réparation & restauration (atelier), entretien & soin "
        "(cuir, soie, métaux), garantie & service à vie, authenticité & "
        "traçabilité, personnalisation & gravure, produits & savoir-faire, "
        "commandes & livraison, retours & échanges, procédures SAV"
    ),
    rewriter_intro=(
        "The whole base is the after-sales & client care knowledge base of a "
        "luxury Maison. Users (client advisors, after-sales specialists, "
        "boutique staff, workshop artisans) ask questions about repair & "
        "restoration, product care, warranty and lifetime service, "
        "authenticity, personalization, orders and client requests."
    ),
    rewriter_keep_terms=(
        "référence produit, numéro de série, numéro de dossier SAV, cuir, "
        "maroquinerie, carré soie, fermoir, boucle, patine, restauration, "
        "entretien, savoir-faire, garantie, service à vie, pièce détachée, "
        "authenticité, traçabilité, personnalisation, gravure, retour, échange"
    ),
    rewriter_examples=(
        '  "how do we restore a scratched leather bag?"\n'
        '    → "restauration cuir rayure maroquinerie atelier patine savoir-faire"\n'
        '  "care instructions for a silk scarf?"\n'
        '    → "entretien carré soie nettoyage soin conservation conseils"\n'
        '  "how to repair a broken clasp on a bag?"\n'
        '    → "réparation fermoir boucle atelier pièce détachée délai prise en charge"\n'
        '  "is this piece covered for a lifetime repair?"\n'
        '    → "garantie service à vie réparation couverture conditions prise en charge"\n'
        '  "client wants to authenticate a vintage item"\n'
        '    → "authenticité traçabilité numéro série vérification provenance"\n'
        '  "engraving options for a leather good?"\n'
        '    → "personnalisation gravure marquage cuir options délai"'
    ),
    resolver_domain_map=(
        "- 'réparation', 'restauration', 'atelier', 'rayure', 'patine', "
        "'fermoir', 'boucle', 'pièce détachée' → réparation / restauration\n"
        "- 'entretien', 'soin', 'nettoyage', 'conservation', 'cuir', 'soie', "
        "'métal', 'conseils' → entretien / soin\n"
        "- 'garantie', 'service à vie', 'couverture', 'prise en charge' → garantie\n"
        "- 'authenticité', 'traçabilité', 'provenance', 'numéro de série', "
        "'vérification', 'contrefaçon' → authenticité\n"
        "- 'personnalisation', 'gravure', 'marquage', 'sur-mesure', "
        "'monogramme' → personnalisation\n"
        "- 'référence', 'produit', 'modèle', 'collection', 'savoir-faire', "
        "'matière' → produits\n"
        "- 'commande', 'livraison', 'expédition', 'suivi' → commandes\n"
        "- 'retour', 'échange', 'remboursement' → retours\n"
        "- 'procédure', 'politique', 'règle interne' → procédures"
    ),
    # No cross-cutting dashboards in the client-care model.
    boundary_rules=(
        # A repair covered by lifetime service spans warranty AND workshop.
        (("garantie réparation", "réparation sous garantie", "service à vie",
          "prise en charge"),
         ("garantie", "reparation", "réparation", "restauration")),
        # Care of a material often overlaps with restoration guidance.
        (("entretien cuir", "soin cuir", "entretien soie", "conservation"),
         ("entretien", "soin", "reparation", "réparation", "restauration")),
    ),
    users_db_path="ui/data/hermes_users.db",
    roles=("responsable_sav", "conseiller_client", "artisan"),
    default_role="conseiller_client",
    role_keywords={
        "responsable_sav": None,   # full access
        "conseiller_client": (
            "garantie", "retour", "commande", "produit",
            "procedure", "authenticite",
        ),
        "artisan": (
            "reparation", "restauration", "entretien",
            "soin", "personnalisation",
        ),
    },
)


# ── Profile: Indica (city hall / mairies) — legacy content, preserved ──────────
_INDICA = BrandProfile(
    name="Indica",
    domain="a city hall (single commune)",
    subtitle="Knowledge base for the city hall",
    description=(
        "Indica indexes the city hall's internal documents — finances, HR, "
        "patrimony, maintenance, business records, indicators — and lets the "
        "mayor, deputy mayors, accountants, HR officers and agents find answers "
        "in plain language, with sourced citations."
    ),
    drive_root_folder="Rag_indica",
    duckdb_path="hybrid/data/hybrid.duckdb",
    theme=Theme(
        accent="#4285F4",
        accent_light="#EBF3FD",
        hybrid="#34A853",
        user="#F8F9FA",
        tool="#FFF3CD",
        error="#F8D7DA",
    ),
    audience="mayors, deputy mayors, accountants, HR officers, agents",
    scope="a French city hall (a single commune)",
    domain_examples=(
        "financière & comptable, RH, patrimoine-maintenance-énergie, "
        "métier divers, divers (rapports administratifs), "
        "tableaux de pilotage-suivi"
    ),
    rewriter_intro=(
        "The whole base belongs to ONE commune. Users (mayors, deputy mayors, "
        "accountants, HR officers, agents) ask questions about internal data — "
        "finances, RH, patrimoine, maintenance, métier, indicateurs de "
        "pilotage, rapports administratifs, etc."
    ),
    rewriter_keep_terms=(
        "budget, dotation, subvention, marché public, délibération, arrêté, "
        "contrat, agent, statut, équipement, parcelle"
    ),
    rewriter_examples=(
        '  "what\'s our debt level for 2024?"\n'
        '    → "endettement encours dette 2024 capacité désendettement"\n'
        '  "how many fonctionnaires titulaires do we have?"\n'
        '    → "effectif fonctionnaires titulaires statut RH agents"\n'
        '  "show me the contract with Veolia"\n'
        '    → "contrat Veolia marché public prestation eau assainissement"\n'
        '  "when does the gym roof need to be replaced?"\n'
        '    → "gymnase toiture rénovation patrimoine échéance maintenance"'
    ),
    resolver_domain_map=(
        "- 'budget', 'dépenses', 'recettes', 'dotation', 'subvention', "
        "'compte administratif', 'fiscalité', 'trésor' → finance / comptable\n"
        "- 'effectifs', 'masse salariale', 'agents', 'fonctionnaires', "
        "'formation', 'paie', 'absentéisme', 'congés' → RH / personnel\n"
        "- 'bâtiment', 'équipement', 'inventaire', 'maintenance', "
        "'travaux', 'énergie', 'parcelle', 'voirie' → patrimoine / maintenance\n"
        "- 'matériel informatique', 'véhicules', 'opérationnel', "
        "'service', 'délégation' → métier divers\n"
        "- 'rapport', 'délibération', 'arrêté', 'orientation', "
        "'égalité' → divers / rapports administratifs\n"
        "- 'indicateurs', 'tableau de bord', 'pilotage', 'mandat' → pilotage / suivi"
    ),
    pilotage_labels=(
        "pilotage", "tableaux de pilotage", "tableaux-de-pilotage",
        "tableau de pilotage", "indicateurs", "suivi",
    ),
    boundary_rules=(
        (("charges patronales", "cotisations sociales", "masse salariale brute",
          "masse salariale", "salaires bruts", "salaires nets", "primes",
          "ijss", "taxe sur salaires"),
         ("rh", "personnel", "finance", "comptable", "financière", "financier")),
        (("travaux énergie", "rénovation thermique", "énergétique"),
         ("patrimoine", "maintenance", "énergie", "finance", "comptable")),
    ),
    users_db_path="ui/data/users.db",   # legacy store — keep existing accounts
    roles=("maire", "adjoint", "comptable", "rh", "agent"),
    default_role="agent",
    role_keywords={
        "maire": None,
        "adjoint": None,
        "comptable": ("finance", "compta", "budget", "tresor"),
        "rh": ("rh", "personnel", "ressources humaines", "ressources_humaines"),
        "agent": ("metier", "service"),
    },
)


# ── Shared domain fragments: customer care (Lacoste + Activate) ───────────────
# Lacoste (client) and Activate (agency house brand) share the same customer-
# care domain; only identity + theme + data store differ. Keep the domain
# fragments here once, reuse in both profiles.
_CC_AUDIENCE = "customer care advisors"
_CC_SCOPE = "a customer care service"
_CC_DOMAIN_EXAMPLES = (
    "retours & SAV, produits & tailles, livraison & commandes, garantie, "
    "RH, marketing, legal, finance"
)
_CC_REWRITER_INTRO = (
    "The whole base is a customer care knowledge base. Advisors type customer "
    "questions (often relayed as-is) about products, sizing, returns, delivery, "
    "orders and warranty, and need a ready-to-relay answer."
)
_CC_REWRITER_KEEP_TERMS = (
    "product names, sizes, references, model names, return, refund, exchange, "
    "delivery, order status, warranty, sizing, SLA"
)
_CC_REWRITER_EXAMPLES = (
    '  "customer wants to return a polo bought 3 weeks ago"\n'
    '    → "return policy polo 3 weeks delay conditions refund"\n'
    '  "what size should I recommend for someone who wears M in slim fit?"\n'
    '    → "slim fit sizing guide medium size conversion size up"\n'
    '  "is the ConnectWatch waterproof?"\n'
    '    → "ConnectWatch water resistance rating specifications"\n'
    '  "how long is the warranty on leather goods?"\n'
    '    → "leather goods warranty duration conditions"\n'
    '  "order 3 days still in preparation"\n'
    '    → "order preparation time SLA 3 days delay escalation warehouse"\n'
    '  "customer received wrong item"\n'
    '    → "wrong item received error shipping claim exchange procedure"'
)
_CC_RESOLVER_DOMAIN_MAP = (
    "- 'return', 'refund', 'exchange', 'rétractation', 'SAV', 'réclamation' → retours / sav\n"
    "- 'size', 'sizing', 'fit', 'taille', 'référence produit', 'model' → produits / tailles\n"
    "- 'delivery', 'shipping', 'order', 'tracking', 'livraison', 'commande' → livraison / commandes\n"
    "- 'warranty', 'garantie', 'guarantee' → garantie\n"
    "- 'HR', 'ressources humaines', 'congés', 'paie' → rh\n"
    "- 'marketing', 'campaign', 'promotion', 'campagne' → marketing\n"
    "- 'legal', 'contract', 'juridique', 'contrat' → legal\n"
    "- 'finance', 'invoice', 'facture', 'budget' → finance"
)
# Customer-care roles (shared by Lacoste + Activate).
_CC_ROLES = ("admin", "superviseur", "conseiller", "manager")
_CC_DEFAULT_ROLE = "conseiller"
_CC_ROLE_KEYWORDS: dict[str, tuple[str, ...] | None] = {
    "admin": None,           # full access
    "superviseur": None,     # full access
    "conseiller": (          # customer-facing topics only
        "retour", "sav", "produit", "taille",
        "livraison", "commande", "garantie",
    ),
    "manager": (             # internal business functions
        "rh", "marketing", "legal", "finance",
    ),
}


# ── Profile: Lacoste (customer care — client instance) ─────────────────────────
_LACOSTE = BrandProfile(
    name="AI for Customer Care",
    domain="customer care",
    subtitle="Workshop Chatbot Knowledge Base",
    description=(
        "AI for Customer Care indexes the internal customer care knowledge base "
        "— products, sizing, returns & SAV, delivery, orders and warranty — and "
        "lets advisors find ready-to-relay answers with sourced citations."
    ),
    drive_root_folder="RAG",
    duckdb_path="Activate_db/lacoste_hybrid.duckdb",
    theme=Theme(
        accent="#006A4E",       # Lacoste dark green
        accent_light="#E6F4ED",  # light green tint
        hybrid="#00A651",       # Lacoste bright green
        user="#F5FAF7",         # soft green-white
        tool="#E6F4ED",
        error="#F8D7DA",
    ),
    audience=_CC_AUDIENCE,
    scope=_CC_SCOPE,
    domain_examples=_CC_DOMAIN_EXAMPLES,
    rewriter_intro=_CC_REWRITER_INTRO,
    rewriter_keep_terms=_CC_REWRITER_KEEP_TERMS,
    rewriter_examples=_CC_REWRITER_EXAMPLES,
    resolver_domain_map=_CC_RESOLVER_DOMAIN_MAP,
    # No cross-cutting dashboards / boundary rules in the customer-care model.
    users_db_path="ui/data/lacoste_users.db",
    roles=_CC_ROLES,
    default_role=_CC_DEFAULT_ROLE,
    role_keywords=_CC_ROLE_KEYWORDS,
)


# ── Profile: Activate (customer care — agency house brand) ─────────────────────
_ACTIVATE = BrandProfile(
    name="Activate AI",
    domain="customer care",
    subtitle="Customer Care Knowledge Base",
    description=(
        "Activate AI indexes the internal customer care knowledge base — "
        "products, sizing, returns & SAV, delivery, orders and warranty — and "
        "lets advisors find ready-to-relay answers with sourced citations."
    ),
    drive_root_folder="RAG",
    duckdb_path="Activate_db/hybrid.duckdb",
    theme=Theme(              # house palette (neutral blue) — tune to charter
        accent="#4285F4",
        accent_light="#EBF3FD",
        hybrid="#34A853",
        user="#F8F9FA",
        tool="#FFF3CD",
        error="#F8D7DA",
    ),
    audience=_CC_AUDIENCE,
    scope=_CC_SCOPE,
    domain_examples=_CC_DOMAIN_EXAMPLES,
    rewriter_intro=_CC_REWRITER_INTRO,
    rewriter_keep_terms=_CC_REWRITER_KEEP_TERMS,
    rewriter_examples=_CC_REWRITER_EXAMPLES,
    resolver_domain_map=_CC_RESOLVER_DOMAIN_MAP,
    users_db_path="ui/data/activate_users.db",
    roles=_CC_ROLES,
    default_role=_CC_DEFAULT_ROLE,
    role_keywords=_CC_ROLE_KEYWORDS,
)


# ── Registry & active selection ───────────────────────────────────────────────
PROFILES: dict[str, BrandProfile] = {
    "hermes": _HERMES,
    "indica": _INDICA,
    "lacoste": _LACOSTE,
    "activate": _ACTIVATE,
}

_DEFAULT_PROFILE = "hermes"

ACTIVE: BrandProfile = PROFILES.get(
    os.getenv("BRAND_PROFILE", _DEFAULT_PROFILE).strip().lower(),
    PROFILES[_DEFAULT_PROFILE],
)


# ── Back-compat flat constants (existing imports keep working) ─────────────────
NAME = ACTIVE.name
DOMAIN = ACTIVE.domain
SUBTITLE = ACTIVE.subtitle
DESCRIPTION = ACTIVE.description
AGENT_NAME = ACTIVE.agent_name
DRIVE_ROOT_FOLDER = ACTIVE.drive_root_folder
DUCKDB_PATH = ACTIVE.duckdb_path
USERS_DB_PATH = ACTIVE.users_db_path
ROLES = ACTIVE.roles
DEFAULT_ROLE = ACTIVE.default_role
ROLE_KEYWORDS = ACTIVE.role_keywords
