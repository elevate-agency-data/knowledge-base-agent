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
    font: str = "sans serif"           # Streamlit theme.font: sans serif|serif|monospace
    # Typography. Streamlit ≥1.5x accepts a full CSS font stack here, not just
    # the three generic keywords, so a brand can name real faces. Stacks are
    # system-only on purpose: no webfont request, so the app keeps working
    # offline and behind a locked-down network.
    #   font_body    : running text
    #   font_heading : titles (same family by default — differentiate by size
    #                  and spacing rather than by loading a second face)
    #   font_code    : marks, references, scores (the poinçon face)
    #   base_font_size: root size in px; luxury editorial reads slightly larger
    font_body: str = ""
    font_heading: str = ""
    font_code: str = ""
    base_font_size: int = 0


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
    #   role_audience : maps a business role to the chunk-level ``audience_role``
    #     values it may see (None = full access). This is the atelier RBAC —
    #     finer than role_keywords (which gates whole indexes by name). When a
    #     role is absent here, the code falls back to role_keywords.
    role_audience: dict[str, tuple[str, ...] | None] = field(default_factory=dict)

    # 5. Branding assets & copy
    #   logo_path : project-relative path to the brand logo (png/svg). Empty →
    #               a styled text wordmark is shown instead. Supply the real
    #               asset per brand; never ship a placeholder trademark.
    #   headline  : Home page display line. Empty → falls back to subtitle.
    logo_path: str = ""
    headline: str = ""

    # 6. "How RAG Works" teaching page — the fallback document shown when no PDF
    #    is uploaded, and the chunks/query used for the live embedding demo.
    #    Keep the text in the brand's own domain: the page is a client-facing
    #    explainer, so a leather-care guide under a customer-care brand jars.
    #    demo_chunks entries are (label, text); demo_query is the question the
    #    chunks are ranked against.
    #    demo_tags are the business tags displayed under the 4 sliced chunks,
    #    following the docs/business_metadata.md convention (facet:value).
    demo_doc_title: str = "Document de référence"
    demo_doc_text: str = ""
    demo_chunks: tuple[tuple[str, str], ...] = ()
    demo_query: str = ""
    demo_tags: tuple[tuple[str, ...], ...] = ()
    #    demo_indexes drive the "vector store" chapter: one entry per business
    #    index/collection as (display_name, illustrative_vector_count). They show
    #    that a store is partitioned into domain indexes, not one flat blob.
    demo_indexes: tuple[tuple[str, int], ...] = ()
    #    demo_query_indexes: which demo_indexes entries the demo_query is routed
    #    to (0-based). The search chapter lights these up and dims the rest, to
    #    show that a query hits only the relevant domain indexes.
    demo_query_indexes: tuple[int, ...] = (0, 1)

    # 7. SAV image analysis (vision tools) — the brand's own vocabulary, fed to
    #    the vision model so it names things the way the Maison does instead of
    #    inventing generic labels. Leave empty to disable the vision tools for
    #    a brand: the tool then reports that this profile has no SAV taxonomy.
    #      vision_product_lines : the product families the model must choose from
    #      vision_materials     : named materials it should recognise
    #      vision_damage_types  : the damage vocabulary used by the workshop
    #      vision_components    : detachable/replaceable parts — drives the
    #                             "what can be salvaged / reused" assessment
    vision_product_lines: tuple[str, ...] = ()
    vision_materials: tuple[str, ...] = ()
    vision_damage_types: tuple[str, ...] = ()
    vision_components: tuple[str, ...] = ()

    @property
    def has_vision(self) -> bool:
        """True when this brand defines an SAV image-analysis taxonomy."""
        return bool(self.vision_product_lines or self.vision_damage_types)

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
    # ── User-facing copy (French: the app is used in French) ───────────────
    # The prompt fragments further down stay in English — they instruct the
    # model, which is separately told to answer in the user's own language.
    subtitle="Base de connaissance du service après-vente",
    description=(
        "Tout ce que la Maison sait de l'après-vente, réuni au même endroit : "
        "réparation et restauration, entretien du cuir, de la soie et des "
        "métaux, garantie, authenticité, personnalisation, commandes et "
        "retours. Posez votre question comme vous la poseriez à un collègue — "
        "la réponse arrive avec les documents dont elle vient."
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
        font="serif",           # classic serif — luxury register
        # Humanist old-style serif: warmer and less newspapery than Georgia,
        # and present on both macOS (Iowan) and Windows (Palatino Linotype).
        font_body=('"Iowan Old Style", "Palatino Linotype", Palatino, '
                   '"Book Antiqua", Georgia, serif'),
        font_heading=('"Iowan Old Style", "Palatino Linotype", Palatino, '
                      '"Book Antiqua", Georgia, serif'),
        font_code=('ui-monospace, "SF Mono", "JetBrains Mono", '
                   '"IBM Plex Mono", Consolas, monospace'),
        base_font_size=17,
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
    role_audience={
        # user role -> chunk audience_role values it may see (None = full)
        "responsable_sav": None,                       # sees everything
        "conseiller_client": ("conseiller", "expert_authentification"),
        "artisan": ("artisan",),
    },
    logo_path="ui/assets/hermes_logo.png",   # drop the official asset here
    headline="Le savoir-faire de la Maison, à portée de question.",
    demo_doc_title="Guide d'entretien — Maroquinerie",
    demo_doc_text=(
        "Guide d'entretien — Maroquinerie\n\n"
        "Cuir grainé (Togo, Clémence)\n"
        "Époussetez régulièrement à l'aide d'un chiffon doux et sec. Évitez toute "
        "exposition prolongée au soleil, à la chaleur et à l'humidité, qui peuvent "
        "altérer la teinte et assouplir la structure du cuir.\n\n"
        "En cas de tache, tamponnez délicatement sans frotter. N'appliquez ni "
        "solvant ni produit gras : ils pénètrent la fibre et laissent une auréole.\n\n"
        "Fermoirs et pièces métalliques\n"
        "Essuyez les parties plaquées avec un linge doux. Le contact prolongé avec "
        "l'eau, les parfums ou les cosmétiques ternit la finition.\n\n"
        "Garantie et prise en charge\n"
        "Chaque pièce peut être confiée en boutique pour un diagnostic. L'atelier "
        "évalue l'usure, propose une réparation ou une restauration, et communique "
        "un délai de prise en charge. Le suivi du dossier est accessible en boutique."
    ),
    demo_chunks=(
        ("chunk 1", "Époussetez le cuir grainé avec un chiffon doux et sec. "
                    "Évitez le soleil et l'humidité."),
        ("chunk 2", "En cas de tache sur le cuir, tamponnez délicatement "
                    "sans frotter ni solvant."),
        ("chunk 3", "La pièce est confiée en boutique ; l'atelier évalue "
                    "l'usure et le délai de prise en charge."),
    ),
    demo_query="comment entretenir le cuir ?",
    demo_tags=(
        ("matiere:cuir", "demande:entretien"),
        ("matiere:cuir", "demande:tache"),
        ("produit:fermoir", "matiere:metal"),
        ("demande:garantie", "cible:conseiller"),
    ),
    demo_indexes=(
        ("Réparation & restauration", 1280),
        ("Entretien & soin", 940),
        ("Garantie", 610),
        ("Authenticité", 430),
        ("Personnalisation", 380),
        ("Produits & savoir-faire", 1120),
    ),
    demo_query_indexes=(0, 1),   # entretenir le cuir -> Réparation + Entretien
    vision_product_lines=(
        "maroquinerie (sac, pochette, portefeuille, ceinture)",
        "soie & textile (carré, twilly, écharpe, cravate)",
        "horlogerie (montre, bracelet de montre)",
        "bijouterie (bracelet, bague, collier, boucles d'oreilles)",
        "art de vivre (porcelaine, cristal, plateau, objet de bureau)",
        "souliers",
    ),
    vision_materials=(
        "cuir Togo", "cuir Clémence", "cuir Epsom", "cuir Swift",
        "cuir Barénia", "box-calf", "chèvre Mysore", "cuir exotique",
        "twill de soie", "cachemire",
        "métal plaqué or", "métal palladié", "or", "argent",
        "porcelaine", "cristal", "bois",
    ),
    vision_damage_types=(
        "rayure", "éraflure", "usure des angles", "usure des arêtes",
        "couture lâchée ou rompue", "déchirure", "perforation",
        "tache", "auréole", "décoloration", "teinte passée",
        "déformation", "affaissement de structure", "cuir sec ou craquelé",
        "oxydation du métal", "plaquage usé", "pièce manquante",
        "mécanisme grippé ou cassé", "accroc", "roulotté défait",
        "verre rayé ou fêlé", "éclat", "fêlure",
    ),
    vision_components=(
        "fermoir", "boucle", "cadenas", "clochette", "clés",
        "sangle", "bandoulière", "poignée", "anse",
        "zip / fermeture éclair", "mousqueton", "pieds de sac",
        "doublure", "garniture intérieure", "poche intérieure",
        "boucle ardillon", "passant", "rivet", "œillet",
        "bracelet de montre", "boîtier", "mouvement", "verre", "couronne",
        "chaîne", "maillon", "pierre / cabochon",
        "semelle", "talon", "lacets",
    ),
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
    demo_doc_title="Délibération — Tarifs des services municipaux",
    demo_doc_text=(
        "Délibération — Tarifs des services municipaux\n\n"
        "Restauration scolaire\n"
        "Le tarif du repas est calculé selon le quotient familial. Les familles "
        "transmettent leur avis d'imposition en début d'année scolaire ; à "
        "défaut, le tarif plein est appliqué jusqu'à régularisation.\n\n"
        "Piscine municipale\n"
        "L'entrée unitaire et l'abonnement trimestriel sont révisés chaque "
        "année. Un tarif réduit s'applique aux résidents de la commune sur "
        "présentation d'un justificatif de domicile.\n\n"
        "Recouvrement\n"
        "Les factures sont émises mensuellement et payables en ligne ou auprès "
        "du régisseur. Une relance est adressée après trente jours ; passé ce "
        "délai, le dossier est transmis au Trésor public.\n\n"
        "Entrée en vigueur\n"
        "Les nouveaux tarifs s'appliquent au premier jour du mois suivant la "
        "publication de la présente délibération."
    ),
    demo_chunks=(
        ("chunk 1", "Le tarif du repas scolaire est calculé selon le quotient "
                    "familial de la famille."),
        ("chunk 2", "Un tarif réduit s'applique aux résidents sur présentation "
                    "d'un justificatif de domicile."),
        ("chunk 3", "Une relance est adressée après trente jours, puis le "
                    "dossier part au Trésor public."),
    ),
    demo_query="comment est calculé le tarif de la cantine scolaire ?",
    demo_tags=(
        ("domaine:scolaire", "cible:agent"),
        ("domaine:piscine", "demande:tarif"),
        ("domaine:finance", "demande:recouvrement"),
        ("domaine:deliberation", "cible:adjoint"),
    ),
    demo_indexes=(
        ("Finances & comptabilité", 1450),
        ("Ressources humaines", 980),
        ("Patrimoine & énergie", 640),
        ("Pilotage & suivi", 520),
        ("Métier & services", 1130),
        ("Divers & rapports", 410),
    ),
    demo_query_indexes=(0, 4),   # tarif cantine -> Finances + Métier & services
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

# Teaching-page demo document — customer-care register (Lacoste, Activate).
_CC_DEMO_TITLE = "Procédure — Retours et remboursements"
_CC_DEMO_TEXT = (
    "Procédure — Retours et remboursements\n\n"
    "Délai de rétractation\n"
    "Le client dispose de trente jours à compter de la réception pour demander "
    "un retour. Au-delà, seul un geste commercial validé par le superviseur "
    "peut être proposé, sans obligation de reprise.\n\n"
    "État de l'article\n"
    "L'article doit être retourné complet, non porté et muni de ses étiquettes "
    "d'origine. Un article abîmé par l'usage est refusé à réception et "
    "réexpédié au client sans frais.\n\n"
    "Frais de retour\n"
    "Les frais sont offerts en cas d'erreur de préparation ou de produit "
    "défectueux. Dans les autres cas, ils restent à la charge du client et "
    "sont déduits du remboursement.\n\n"
    "Remboursement\n"
    "Le remboursement est déclenché après contrôle en entrepôt et intervient "
    "sous cinq à sept jours ouvrés sur le moyen de paiement d'origine. Le "
    "conseiller communique le numéro de dossier pour le suivi."
)
_CC_DEMO_CHUNKS = (
    ("chunk 1", "Le client dispose de trente jours après réception pour "
                "demander un retour."),
    ("chunk 2", "L'article doit être complet, non porté et muni de ses "
                "étiquettes d'origine."),
    ("chunk 3", "Le remboursement intervient sous cinq à sept jours ouvrés "
                "après contrôle en entrepôt."),
)
_CC_DEMO_QUERY = "combien de temps a le client pour demander un retour ?"
_CC_DEMO_TAGS = (
    ("demande:retour", "cible:conseiller"),
    ("demande:retour", "produit:article"),
    ("demande:frais", "cible:conseiller"),
    ("demande:remboursement", "cible:superviseur"),
)
_CC_DEMO_INDEXES = (
    ("Retours & remboursements", 1040),
    ("Commandes", 1310),
    ("Livraison", 720),
    ("Produits & tailles", 1180),
    ("Fidélité & compte", 560),
    ("Réclamations", 480),
)


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
    demo_doc_title=_CC_DEMO_TITLE,
    demo_doc_text=_CC_DEMO_TEXT,
    demo_chunks=_CC_DEMO_CHUNKS,
    demo_query=_CC_DEMO_QUERY,
    demo_tags=_CC_DEMO_TAGS,
    demo_indexes=_CC_DEMO_INDEXES,
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
    # Historical store — hosts the house indexes (elev8, …) alongside the
    # client ones. Kept as-is so the existing knowledge base stays reachable.
    duckdb_path="Activate_db/lacoste_hybrid.duckdb",
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
    users_db_path="ui/data/users.db",   # legacy store — keep existing accounts
    roles=_CC_ROLES,
    default_role=_CC_DEFAULT_ROLE,
    role_keywords=_CC_ROLE_KEYWORDS,
    demo_doc_title=_CC_DEMO_TITLE,
    demo_doc_text=_CC_DEMO_TEXT,
    demo_chunks=_CC_DEMO_CHUNKS,
    demo_query=_CC_DEMO_QUERY,
    demo_tags=_CC_DEMO_TAGS,
    demo_indexes=_CC_DEMO_INDEXES,
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
