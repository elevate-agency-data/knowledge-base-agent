"""
How RAG Works — an animated, scroll-through explainer.

A self-contained HTML/CSS/JS story: a REAL PDF page (uploaded, or a realistic
sample rendered with pymupdf) is sliced into chunks before your eyes — the
strips separate, the overlap band lights up, metadata chips attach — then the
chunks become dense embedding vectors, land in the store, and get retrieved.

No knowledge-base backend is touched, so it works with an empty base. Themed
from the active brand accent.
"""

import base64
import json
import math

import path_setup  # noqa: F401
import streamlit as st
from streamlit.components.v1 import html as st_html

from config import APP_TITLE, APP_ICON, LAYOUT
from auth import require_auth
from components.sidebar_auth import render_sidebar_user
from shared.brand import ACTIVE

st.set_page_config(page_title=f"How RAG works — {APP_TITLE}",
                   page_icon=APP_ICON, layout=LAYOUT)

require_auth()

with st.sidebar:
    render_sidebar_user()

_ACCENT = ACTIVE.theme.accent

# Demo corpus — driven by the active brand profile so the explainer speaks the
# brand's own domain. Fallbacks keep the page working for a profile that hasn't
# defined one yet.
_DOC_TITLE = ACTIVE.demo_doc_title or "Document de référence"
_SAMPLE_TEXT = ACTIVE.demo_doc_text or (
    f"{_DOC_TITLE}\n\n"
    "Ce document sert d'exemple à la démonstration. Déposez un PDF pour voir "
    "le découpage s'appliquer à votre propre contenu."
)


@st.cache_data(show_spinner=False)
def _render_pdf_png(data: bytes | None) -> str | None:
    """Render page 1 of a PDF (or a sample doc) to a base64 PNG data URI."""
    try:
        import fitz  # pymupdf
    except Exception:
        try:
            import pymupdf as fitz  # newer import name
        except Exception:
            return None
    try:
        if data:
            doc = fitz.open(stream=data, filetype="pdf")
        else:
            doc = fitz.open()
            page = doc.new_page(width=420, height=560)
            page.insert_textbox(
                fitz.Rect(40, 46, 380, 540), _SAMPLE_TEXT,
                fontsize=11, fontname="helv", lineheight=1.35,
            )
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6))
        png = pix.tobytes("png")
        doc.close()
        return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    except Exception:
        return None


_DEMO_CHUNKS = list(ACTIVE.demo_chunks) or [
    ("chunk 1", "Premier passage du document, découpé à taille fixe."),
    ("chunk 2", "Deuxième passage, qui recouvre partiellement le premier."),
    ("chunk 3", "Troisième passage, porteur de la réponse recherchée."),
]
_DEMO_QUERY = ACTIVE.demo_query or "que dit ce document ?"


_SMALL_DIM = 8  # illustrative "tiny model" — first 8 dims of the real vector


@st.cache_data(show_spinner="Calcul des embeddings…")
def _embed_demo() -> dict | None:
    """Embed demo chunks + query and build two views: full-dim vs tiny-dim.

    Both views carry a 2D PCA projection and the cosine similarity of every
    chunk to the query (the reference). The tiny view truncates the vector to
    ``_SMALL_DIM`` dimensions to show how a smaller embedding degrades the
    structure. Returns None if the model can't load (offline demo).
    """
    try:
        import numpy as np
        from hybrid.embeddings import get_embedding_model
        from hybrid.config import DEFAULT_EMBEDDING_MODEL

        model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
        texts = [t for _, t in _DEMO_CHUNKS] + [_DEMO_QUERY]
        vecs = np.array([model.embed_query(t) for t in texts], dtype=float)
        ref = len(_DEMO_CHUNKS)  # query index (reference)
        labels = [lbl for lbl, _ in _DEMO_CHUNKS] + ["question"]

        def csim(a, b):
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        # Radial layout: the query sits at the centre, each chunk at a FIXED
        # angle (same in both views), radius = semantic distance to the query.
        # Only the radius changes between views, so distances compare directly.
        _CX, _CY, _K = 180.0, 160.0, 340.0
        _ANGLES = [math.radians(a) for a in (152, 24, 286)]  # chunk 0,1,2

        def _view(mat: "np.ndarray") -> dict:
            sims = [round(csim(mat[i], mat[ref]), 3) for i in range(len(mat))]
            pts = []
            for i in range(len(mat)):
                if i == ref:
                    pts.append([_CX, _CY])
                    continue
                r = min(150.0, max(30.0, (1.0 - sims[i]) * _K))
                a = _ANGLES[i] if i < len(_ANGLES) else i
                pts.append([float(round(_CX + r * math.cos(a), 1)),
                            float(round(_CY + r * math.sin(a), 1))])
            return {"dim": int(mat.shape[1]), "pts": pts, "sims": sims}

        # Choose 8 REAL dimensions that make a DIFFERENT chunk the closest to
        # the query than the full-dim winner (favour chunk 3), to illustrate how
        # a poor low-dim embedding mis-ranks. Deterministic search.
        D = vecs.shape[1]
        cand = [tuple(range(s, s + _SMALL_DIM)) for s in range(0, D - _SMALL_DIM, 3)]
        for st in range(1, 60):
            sset = {(i * st) % D for i in range(_SMALL_DIM)}
            if len(sset) == _SMALL_DIM:
                cand.append(tuple(sorted(sset)))
        best, best_score = tuple(range(_SMALL_DIM)), -9.0
        for dims in cand:
            sub = vecs[:, list(dims)]
            s = [csim(sub[i], sub[ref]) for i in range(3)]
            score = s[2] - max(s[0], s[1])       # favour chunk 3 being closest
            if score > best_score:
                best_score, best = score, dims
        small_mat = vecs[:, list(best)]

        full = _view(vecs)
        small = _view(small_mat)

        full_rows, small_rows = [], []
        for i, (label, _) in enumerate(_DEMO_CHUNKS):
            full_rows.append({
                "label": label,
                "preview": [round(float(x), 4) for x in vecs[i][:8]],
                "heat": [round(float(x), 5) for x in vecs[i][:40]],
            })
            small_rows.append({
                "label": label,
                "vals": [round(float(x), 4) for x in small_mat[i]],
                "heat": [round(float(x), 5) for x in small_mat[i]],
            })

        return {
            "model": DEFAULT_EMBEDDING_MODEL,
            "ref": ref,
            "labels": labels,
            "full": {**full, "rows": full_rows},
            "small": {**small, "rows": small_rows},
            "query": {"heat": [round(float(x), 5) for x in vecs[ref][:16]]},
        }
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _pdf_text(data: bytes) -> str:
    """Extract the text of the first pages of an uploaded PDF."""
    try:
        import fitz
    except Exception:
        try:
            import pymupdf as fitz
        except Exception:
            return ""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        parts = [doc[i].get_text() for i in range(min(3, doc.page_count))]
        doc.close()
        return "\n".join(parts)
    except Exception:
        return ""


def _naive_cut(text: str) -> dict | None:
    """Naive fixed-size cut that lands MID-WORD, to show truncation.

    Returns the context around the split and the two halves of the broken word,
    so the UI can highlight the real truncated token from the uploaded document.
    """
    text = " ".join(text.split())
    if len(text) < 80:
        return None
    target = len(text) // 2
    cut = None
    for d in range(80):
        for c in (target + d, target - d):
            if 1 <= c < len(text) - 1 and text[c - 1].isalnum() and text[c].isalnum():
                cut = c
                break
        if cut:
            break
    if not cut:
        return None
    head, tail = text[:cut], text[cut:]
    j = len(head)
    while j > 0 and head[j - 1].isalnum():
        j -= 1
    k = 0
    while k < len(tail) and tail[k].isalnum():
        k += 1
    c1_cut, c2_cut = head[j:], tail[:k]
    if not c1_cut or not c2_cut:
        return None
    return {
        "c1": head[:j][-120:],   # context at the end of chunk 1
        "c1_cut": c1_cut,        # head of the broken word (stays in chunk 1)
        "c2_cut": c2_cut,        # tail of the broken word (starts chunk 2)
        "c2": tail[k:][:120],    # context at the start of chunk 2
    }


# ── PDF source ────────────────────────────────────────────────────────────────

up = st.file_uploader("Déposez un PDF (ou laissez l'exemple)", type=["pdf"])
doc_src = _render_pdf_png(up.getvalue() if up else None)
_naive = _naive_cut(_pdf_text(up.getvalue())) if up else None
_naive_json = json.dumps(_naive) if _naive else "null"

if doc_src is None:
    st.warning("Rendu PDF indisponible (pymupdf). L'animation utilise un gabarit.")
    doc_src = ""  # JS falls back to a drawn page

_emb = _embed_demo()
_emb_json = json.dumps(_emb) if _emb else "null"

# Business tags shown under the 4 sliced chunks — brand-driven, padded/trimmed
# to the 4 slices the animation draws.
_tags = [list(t) for t in (ACTIVE.demo_tags or ())][:4]
while len(_tags) < 4:
    _tags.append(["page 1"])
_tags_json = json.dumps(_tags, ensure_ascii=False)

# Retrieval matches — real cosine similarities to the query when the embedding
# model is available, so the last chapter shows measured numbers, not props.
if _emb:
    _sims = _emb["full"]["sims"]
    _ranked = sorted(
        ((lbl, _sims[i], txt) for i, (lbl, txt) in enumerate(_DEMO_CHUNKS)),
        key=lambda r: r[1], reverse=True,
    )
    _matches = [[f"{lbl} — {txt[:38].rstrip()}…", f"{sim:.2f}"]
                for lbl, sim, txt in _ranked]
else:
    _matches = [[lbl, "—"] for lbl, _ in _DEMO_CHUNKS]
_matches_json = json.dumps(_matches, ensure_ascii=False)

_doc_name = (up.name if up else f"{_DOC_TITLE[:28]}.pdf")
_doc_name_json = json.dumps(_doc_name, ensure_ascii=False)

# Vector-store chapter — the brand's business indexes (name, vector count).
_indexes = [[n, int(c)] for n, c in (ACTIVE.demo_indexes or ())] or [
    ["Domaine A", 900], ["Domaine B", 640], ["Domaine C", 480],
    ["Domaine D", 720], ["Domaine E", 530], ["Domaine F", 410],
]
_indexes_json = json.dumps(_indexes, ensure_ascii=False)

# Which of those indexes the demo query is routed to (clamped to range).
_qidx = [i for i in (ACTIVE.demo_query_indexes or (0, 1)) if 0 <= i < len(_indexes)]
_qidx_json = json.dumps(_qidx or [0])

# Generation chapter — real chunk text, retrieval vs rerank scores, and the
# prompt the chunks get pasted into. demo_chunks[0] is authored as the answer,
# so rerank promotes it; retrieval order is left in document order to show the
# rerank actually reordering.
_gen_sims = (_emb["full"]["sims"][:len(_DEMO_CHUNKS)] if _emb
             else [0.86, 0.79, 0.72][:len(_DEMO_CHUNKS)])
_gen_chunks = [
    {"label": lbl, "text": txt, "retr": round(float(_gen_sims[i]), 2)}
    for i, (lbl, txt) in enumerate(_DEMO_CHUNKS)
]
# rerank: correct chunk first, then the rest by descending similarity
_rerank_order = [0] + sorted(
    [i for i in range(len(_gen_chunks)) if i != 0], key=lambda i: -_gen_sims[i]
)
_rerank_vals = [0.98, 0.64, 0.31, 0.18]
for _rank, _ci in enumerate(_rerank_order):
    _gen_chunks[_ci]["rerank"] = _rerank_vals[min(_rank, len(_rerank_vals) - 1)]
    _gen_chunks[_ci]["rank"] = _rank
# retrieval display order: document order, but ensure it differs from rerank
# order so the reordering is visible (swap first two if identical)
_retr_order = list(range(len(_gen_chunks)))
if _retr_order[:2] == _rerank_order[:2] and len(_retr_order) >= 2:
    _retr_order[0], _retr_order[1] = _retr_order[1], _retr_order[0]
_gen = {
    "query": _DEMO_QUERY,
    "sys": (f"Tu es l'assistant {ACTIVE.name}. Réponds UNIQUEMENT à partir du "
            "contexte ci-dessous. Cite tes sources. Si l'information manque, "
            "dis-le — n'invente rien."),
    "chunks": _gen_chunks,
    "retrOrder": _retr_order,
    "rerankOrder": _rerank_order,
}
_gen_json = json.dumps(_gen, ensure_ascii=False)

_STORY = r"""
<div id="rag-story">
  <button class="fs-btn" id="fsBtn" title="Plein écran">⛶ Plein écran</button>
  <div class="rail"><span data-c="0"></span><span data-c="1"></span><span data-c="2"></span>
    <span data-c="3"></span><span data-c="4"></span><span data-c="5"></span></div>

  <!-- 0 intro + ingestion -->
  <section class="chap" data-i="0">
    <div class="kicker">Pipeline RAG</div>
    <h1>De la page au vecteur, du vecteur à la réponse.</h1>
    <p class="lead">Un document est lu, découpé, vectorisé, indexé, puis retrouvé.
       Faites défiler pour voir chaque étape s'animer. Tout part d'une page —
       celle-ci. (Déposez votre PDF en haut pour l'utiliser.)</p>
    <div class="step" style="margin-top:24px;text-align:center">01 — Ingestion</div>
    <div class="stage" style="margin-top:12px;justify-content:center">
      <div class="pdf" id="pdf1"></div>
    </div>
    <div class="scrollhint">↓ défiler</div>
  </section>

  <!-- 1 chunking : doc → chunks simples (pourquoi) → chunks avec overlap -->
  <section class="chap tall" data-i="1">
    <div class="step">02 — Chunking · Overlap · Métadonnées</div>
    <h2>Du document aux chunks — et pourquoi l'overlap</h2>
    <p class="lead"><b>À gauche</b> le document. <b>Au milieu</b>, la découpe
       simple : une phrase peut être tranchée en plein mot (<span
       style="color:#c0392b">en rouge</span>) — la suite manque. <b>À droite</b>,
       l'overlap répète la zone de coupure (<span style="color:__ACCENT__">en
       orange</span>), avec les métadonnées.</p>
    <div class="stage slice-stage">
      <div class="orig"><div class="pdf" id="pdf2"></div><div class="orig-cap">source</div></div>
      <div class="arrow">→</div>
      <div class="mid" id="ovMid"><div class="col-cap bad">découpe simple — sens coupé</div></div>
      <div class="arrow">→</div>
      <div class="slices-wrap"><div class="col-cap good">avec overlap — continuité</div>
        <div class="slices" id="slices"></div></div>
    </div>
    <button class="replay" id="replay">↻ rejouer</button>
  </section>

  <!-- 2 embeddings : haute dim vs petite dim -->
  <section class="chap tall" data-i="2">
    <div class="step">03 — Vectorisation</div>
    <h2>La dimension dépend du modèle</h2>
    <p class="lead">Des <b>floats</b>, pas des identifiants. À <b>gauche</b>, un
       embedding <b>haute dimension</b> (768) capte finement le sens. À
       <b>droite</b>, un embedding <b>très petit</b> (8) : moins d'information,
       la structure sémantique se dégrade. Dans les deux espaces, la
       <b>question</b> est la référence — on mesure la distance à chaque chunk.</p>
    <div class="model-badge" id="modelBadge"></div>
    <div class="stage embed2">
      <div class="ecol">
        <div class="ecol-h" id="hFull"></div>
        <div class="embeds" id="embFull"></div>
        <div class="space" id="spFull"><div class="space-label">distances à la question — 768 dim</div></div>
      </div>
      <div class="ecol">
        <div class="ecol-h" id="hSmall"></div>
        <div class="embeds" id="embSmall"></div>
        <div class="space" id="spSmall"><div class="space-label">distances à la question — 8 dim</div></div>
      </div>
    </div>
  </section>

  <!-- 3 store : vector store = plusieurs index de domaine -->
  <section class="chap tall" data-i="3">
    <div class="step">04 — Le vector store</div>
    <h2>Un store, plusieurs index</h2>
    <p class="lead">Un <b>vector store</b> n'est pas un sac unique de vecteurs :
       il est découpé en <b>index</b>, un par domaine métier. Chaque vecteur est
       <b>routé</b> vers son index selon sa catégorie. À l'intérieur, un graphe
       <b>HNSW</b> relie chaque vecteur à ses plus proches voisins — la recherche
       saute de voisin en voisin au lieu de tout comparer.</p>

    <div class="vs-wrap">
      <!-- explications à gauche -->
      <aside class="vs-explain">
        <div class="vs-e"><span class="vs-n">1</span>
          <div><b>Vector store</b><br>le conteneur global, une famille d'index
          isolés — l'isolation par domaine (et par client) se fait ici.</div></div>
        <div class="vs-e"><span class="vs-n">2</span>
          <div><b>Index</b><br>une collection de vecteurs d'un même domaine
          (+ leurs métadonnées). On n'interroge que les index utiles.</div></div>
        <div class="vs-e"><span class="vs-n">3</span>
          <div><b>HNSW</b><br>un graphe de voisinage à l'intérieur de l'index :
          recherche approximative en O(log n), pas de balayage complet.</div></div>
      </aside>

      <!-- routage + grille d'index à droite -->
      <div class="vs-main">
        <div class="vs-feed" id="vsFeed"></div>
        <div class="vs-grid" id="vsGrid"></div>
      </div>
    </div>
    <button class="replay" id="replayVs">↻ rejouer le routage</button>
  </section>

  <!-- 4 search : recherche animée à travers les index du store -->
  <section class="chap tall" data-i="4">
    <div class="step">05 — La recherche</div>
    <h2>Une question traverse le store</h2>
    <p class="lead">La question devient un <b>vecteur</b>. Le système ne fouille
       pas tout : il <b>route</b> la requête vers les seuls index pertinents
       (les autres restent <span style="color:#b0aaa3">grisés</span>), puis
       <b>parcourt le graphe HNSW</b> de voisin en voisin jusqu'aux vecteurs
       les plus proches — qui remontent comme résultats.</p>

    <div class="se-query" id="seQuery">
      <span class="se-qtxt">« __QUERY__ »</span>
      <span class="se-qvec" id="seQvec"></span>
    </div>
    <ol class="se-steps" id="seSteps">
      <li data-s="0"><b>Point d'entrée</b><span>on démarre sur un vecteur au hasard du graphe</span></li>
      <li data-s="1"><b>Descente</b><span>on saute vers le voisin le plus proche de la question</span></li>
      <li data-s="2"><b>Vecteur pertinent</b><span>plus aucun voisin n'est plus proche : on s'arrête</span></li>
      <li data-s="3"><b>Voisinage → résultats</b><span>ses voisins immédiats forment les k candidats</span></li>
    </ol>
    <div class="vs-grid" id="seGrid"></div>
    <div class="se-results" id="seResults"></div>
    <button class="replay" id="replaySe">↻ relancer la recherche</button>
  </section>

  <!-- 5 generation : retrieve → rerank → paste into prompt → send -->
  <section class="chap tall" data-i="5">
    <div class="step">06 — Génération</div>
    <h2>Récupérer, coller dans le prompt, envoyer</h2>
    <p class="lead">Pas de magie : le système <b>récupère</b> les chunks,
       les <b>reclasse</b> (rerank), <b>colle leur texte</b> dans le prompt,
       et l'envoie au LLM. Exactement le <b>Ctrl+C / Ctrl+V</b> que tu ferais
       à la main — en automatique.</p>

    <div class="gen">
      <div class="gen-lbl"><span class="gen-n">1</span>Chunks récupérés
        <em id="genPhase">— tri par similarité</em></div>
      <div class="gen-chunks" id="genChunks"></div>

      <div class="gen-flow" id="flowPrompt">↓ &nbsp;copier le texte&nbsp; ↓</div>

      <div class="gen-lbl"><span class="gen-n">2</span>Injection dans le prompt
        <span class="paste-badge" id="pasteBadge">📋 Ctrl+V</span></div>
      <div class="prompt-card" id="promptCard">
        <div class="pc-line pc-sys"><span class="pc-tag">SYSTEM</span>
          <span id="pcSys"></span></div>
        <div class="pc-line pc-ctx"><span class="pc-tag">CONTEXTE</span>
          <div class="pc-ctx-body" id="pcCtx"></div></div>
        <div class="pc-line pc-q"><span class="pc-tag">QUESTION</span>
          <span id="pcQ"></span></div>
      </div>

      <div class="gen-flow" id="flowSend">↓ &nbsp;envoi au LLM&nbsp; ↓</div>

      <div class="gen-lbl"><span class="gen-n">3</span>Réponse générée</div>
      <div class="answer" id="genAnswer"></div>
    </div>
    <button class="replay" id="replayGen">↻ rejouer</button>
  </section>
</div>

<style>
  #rag-story{font-family:inherit;color:#1a1a1a;background:#fff;}
  .fs-btn{position:fixed;top:12px;right:16px;z-index:30;background:#fff;
     border:1px solid #e0dbd5;color:#6a655f;border-radius:20px;padding:6px 14px;
     font-size:.74rem;cursor:pointer;box-shadow:0 8px 20px -12px #00000055;}
  .fs-btn:hover{border-color:__ACCENT__;color:__ACCENT__;}
  #rag-story:fullscreen{overflow-y:auto;background:#fff;}
  #rag-story:-webkit-full-screen{overflow-y:auto;background:#fff;}
  #rag-story h1{font-size:2.4rem;font-weight:600;line-height:1.2;margin:.2em 0;
     max-width:none;white-space:nowrap;}
  #rag-story h2{font-size:1.55rem;font-weight:600;margin:.1em 0 .3em;}
  #rag-story .lead{color:#55504b;line-height:1.75;max-width:62ch;font-size:1rem;}
  .kicker,.step{font-size:.7rem;letter-spacing:.24em;text-transform:uppercase;color:#8f8b86;}
  .step{color:__ACCENT__;}
  .chap{min-height:88vh;padding:8vh 8% 6vh;display:flex;flex-direction:column;
        justify-content:center;opacity:0;transform:translateY(28px);
        transition:opacity .7s,transform .7s;border-top:1px solid #f0edea;}
  .chap.tall{min-height:104vh;}
  .chap.in{opacity:1;transform:none;}
  .scrollhint{margin-top:22px;color:#b7b1aa;font-size:.8rem;letter-spacing:.1em;
        animation:bob 1.6s ease-in-out infinite;}
  @keyframes bob{50%{transform:translateY(6px);}}
  .rail{position:fixed;left:14px;top:50%;transform:translateY(-50%);display:flex;
        flex-direction:column;gap:10px;z-index:5;}
  .rail span{width:8px;height:8px;border-radius:50%;background:#e3ded9;transition:.3s;}
  .rail span.on{background:__ACCENT__;transform:scale(1.4);}
  .stage{margin-top:26px;display:flex;justify-content:center;align-items:flex-start;}

  /* pdf page */
  .pdf{width:260px;border:1px solid #e0dbd5;border-radius:6px;overflow:hidden;
       box-shadow:0 20px 44px -26px #00000055;background:#fff;}
  .pdf img{display:block;width:100%;}
  .pdf-fallback{width:260px;height:346px;background:
     repeating-linear-gradient(#fff,#fff 22px,#f4f1ec 22px,#f4f1ec 30px);}

  /* slicing scene */
  .slice-stage{gap:16px;align-items:flex-start;flex-wrap:wrap;}
  .orig{display:flex;flex-direction:column;align-items:center;gap:6px;}
  .orig-cap{font-size:.6rem;letter-spacing:.14em;text-transform:uppercase;color:#b7b1aa;}
  .arrow{color:#d8d2cb;font-size:1.4rem;margin-top:120px;}
  .mid{width:248px;}
  .slices-wrap{display:flex;flex-direction:column;}
  .col-cap{font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px;}
  .col-cap.bad{color:#c0392b;}.col-cap.good{color:__ACCENT__;}
  .slices{width:264px;}
  .slice-row{transition:margin .6s cubic-bezier(.2,.8,.2,1),transform .6s;}
  .slices.cut .slice-row{margin:12px 0;}
  .slices.cut .slice-row:nth-child(odd){transform:translateX(-5px);}
  .slices.cut .slice-row:nth-child(even){transform:translateX(5px);}
  .slice{position:relative;width:260px;overflow:hidden;border:1px solid transparent;
     border-radius:5px;transition:border-color .6s,box-shadow .6s;}
  .slice img{display:block;width:260px;}
  .slices.cut .slice{border-color:#e4dfda;border-left:3px solid __ACCENT__;
     box-shadow:0 10px 26px -20px #00000055;background:#fff;}
  .ov-band{position:absolute;left:0;right:0;background:__ACCENT__;opacity:0;
     mix-blend-mode:multiply;transition:opacity .5s;pointer-events:none;}
  .slices.cut .ov-band{opacity:.30;}
  /* metadata BELOW each slice — never over the image, never clipped */
  .meta{opacity:0;transform:translateY(4px);transition:.5s;margin:5px 0 2px;
     display:flex;flex-wrap:wrap;}
  .slices.meta-in .meta{opacity:1;transform:none;}
  .meta .chip{font-size:.58rem;letter-spacing:.02em;background:#f4f1ec;color:#6a655f;
     border-radius:3px;padding:1px 6px;margin:0 4px 4px 0;}
  .meta .chip.tag{background:__ACCENT__14;color:__ACCENT__;}
  .ov-legend{display:flex;align-items:center;gap:7px;font-size:.62rem;color:#8f8b86;
     margin-top:10px;opacity:0;transition:.5s;}
  .slices.cut .ov-legend{opacity:1;}
  .ov-legend .sw{width:22px;height:9px;border-radius:2px;background:__ACCENT__;opacity:.5;}
  .cutline{position:absolute;left:-4px;right:-4px;height:2px;background:__ACCENT__;
     transform:scaleX(0);transform-origin:left;z-index:2;}
  .slices.slicing .cutline{animation:cut .5s ease forwards;}
  @keyframes cut{to{transform:scaleX(1);}}
  .replay{margin:20px auto 0;background:none;border:1px solid #e0dbd5;color:#6a655f;
     border-radius:20px;padding:6px 16px;font-size:.75rem;cursor:pointer;}
  .replay:hover{border-color:__ACCENT__;color:__ACCENT__;}

  /* overlap comparison */
  .ov-compare{gap:26px;align-items:flex-start;flex-wrap:wrap;}
  .ovcol{width:320px;}
  .ovcol-h{font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#8f8b86;
     margin-bottom:10px;}
  .ovcol-h .bad{color:#c0392b;}.ovcol-h .good{color:__ACCENT__;}
  .ovcard{border:1px solid #e4dfda;border-left:3px solid #cfc9c2;border-radius:6px;
     background:#fff;padding:11px 13px;margin-bottom:10px;font-size:.84rem;line-height:1.7;
     color:#3d3a36;opacity:0;transform:translateY(10px);transition:.5s;}
  .in .ovcard{opacity:1;transform:none;}
  .ovcard .lbl{font-size:.58rem;letter-spacing:.1em;text-transform:uppercase;
     color:#a49e97;display:block;margin-bottom:5px;}
  .cut-word{background:#c0392b18;color:#c0392b;border-bottom:2px solid #c0392b;
     border-radius:2px 2px 0 0;padding:0 2px;font-weight:600;
     opacity:0;transition:.4s;}
  .lit .cut-word{opacity:1;}
  .rep{background:__ACCENT__1f;color:#8a4a12;border-radius:3px;padding:0 3px;
     box-shadow:inset 0 -2px 0 __ACCENT__;opacity:0;transition:.4s;}
  .lit .rep{opacity:1;}
  .ovnote{font-size:.68rem;margin-top:2px;opacity:0;transition:.5s;}
  .lit .ovnote{opacity:1;}
  .ovnote.bad{color:#c0392b;}.ovnote.good{color:__ACCENT__;}

  /* embeddings */
  .model-badge{margin:14px 0 2px;font-family:monospace;font-size:.72rem;color:#6a655f;}
  .model-badge b{color:__ACCENT__;}
  .embed-stage{gap:26px;flex-wrap:wrap;}
  .embed2{gap:90px;flex-wrap:wrap;align-items:flex-start;justify-content:center;}
  .ecol{width:400px;opacity:0;transform:translateY(16px);transition:.6s;}
  .in .ecol{opacity:1;transform:none;}
  .ecol:nth-child(2){transition-delay:.15s;}
  .ecol-h{font-family:monospace;font-size:.72rem;color:#6a655f;margin-bottom:10px;}
  .ecol-h b{color:__ACCENT__;}
  .space svg{position:absolute;inset:0;overflow:visible;pointer-events:none;}
  .pca-dot{position:absolute;width:14px;height:14px;border-radius:50%;
     transform:translate(-50%,-50%);background:__ACCENT__;transition:.6s;}
  .pca-dot.far{background:#8a6d3b;}
  .pca-dot.ref{width:19px;height:19px;background:#1a1a1a;
     box-shadow:0 0 0 5px __ACCENT__33;}
  .pca-name{position:absolute;font-size:.62rem;color:#8f8b86;
     transform:translate(-50%,-50%);white-space:nowrap;}
  .embeds{display:flex;flex-direction:column;gap:18px;}
  .emb{opacity:0;transform:translateX(-12px);transition:.6s;}
  .in .emb{opacity:1;transform:none;}
  .emb-head{display:flex;align-items:baseline;gap:10px;margin-bottom:4px;}
  .emb-head .txt{font-size:.74rem;color:#8f8b86;font-weight:600;}
  .emb-head .dim{font-family:monospace;font-size:.64rem;color:#8f8b86;}
  .floats{font-family:monospace;font-size:.68rem;color:#3d3a36;margin-bottom:6px;
     white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:400px;}
  .floats .pos{color:__ACCENT__;}.floats .neg{color:#2f6fdc;}.floats .mut{color:#b7b1aa;}
  .vec{display:flex;gap:2px;}
  .cell{width:9px;height:26px;border-radius:1px;}
  .dim{font-family:monospace;font-size:.66rem;color:#8f8b86;}
  .space{width:360px;height:320px;border:1px solid #eee7e0;border-radius:8px;position:relative;
     background:linear-gradient(0deg,#fbf8f5,#fff);margin-top:6px;}
  .space-label{position:absolute;top:8px;left:10px;font-size:.6rem;color:#b7b1aa;}
  .dot{position:absolute;width:13px;height:13px;border-radius:50%;
     box-shadow:0 0 0 4px __ACCENT__22;transform:scale(0);transition:.6s;}
  .in .dot{transform:scale(1);}
  .d0{background:__ACCENT__;left:60px;top:70px;}.d1{background:__ACCENT__;left:92px;top:92px;}
  .d2{background:#8a6d3b;left:150px;top:44px;}

  /* vector store */
  .vs-wrap{display:flex;gap:26px;align-items:flex-start;width:100%;
    max-width:960px;margin:8px auto 0;flex-wrap:wrap;}
  .vs-explain{flex:0 0 240px;display:flex;flex-direction:column;gap:14px;}
  .vs-e{display:flex;gap:11px;align-items:flex-start;font-size:.78rem;
    line-height:1.45;color:#55504b;opacity:0;transform:translateX(-10px);
    transition:.5s;}
  .in .vs-e{opacity:1;transform:none;}
  .vs-e:nth-child(2){transition-delay:.12s;}
  .vs-e:nth-child(3){transition-delay:.24s;}
  .vs-n{flex:0 0 22px;height:22px;border-radius:50%;background:__ACCENT__;
    color:#fff;font-size:.72rem;font-weight:700;display:flex;
    align-items:center;justify-content:center;margin-top:1px;}
  .vs-e b{color:#2b2622;}
  .vs-main{flex:1 1 380px;min-width:320px;}
  /* incoming vectors feed */
  .vs-feed{position:relative;height:34px;margin-bottom:10px;}
  .vs-tok{position:absolute;top:6px;width:22px;height:14px;border-radius:3px;
    background:linear-gradient(90deg,__ACCENT__,#d95f16);opacity:0;
    box-shadow:0 1px 3px #0002;}
  .vs-tok.fly{animation:vsfly .9s ease-in forwards;}
  @keyframes vsfly{0%{opacity:0;transform:translateY(-4px) scale(.9);}
    15%{opacity:1;}100%{opacity:0;transform:translate(var(--dx),var(--dy)) scale(.4);}}
  /* index cards grid */
  .vs-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
  @media(max-width:620px){.vs-grid{grid-template-columns:repeat(2,1fr);}}
  .vs-card{border:1px solid #e6e1db;border-top:2px solid __ACCENT__;
    border-radius:7px;background:#fff;padding:9px 10px 8px;opacity:0;
    transform:translateY(12px);transition:.5s;}
  .in .vs-card{opacity:1;transform:none;}
  .vs-card.pulse{box-shadow:0 0 0 3px __ACCENT__22;}
  .vs-ch{display:flex;justify-content:space-between;align-items:baseline;
    gap:6px;margin-bottom:6px;}
  .vs-ci{font-size:.72rem;font-weight:600;color:#2b2622;line-height:1.2;}
  .vs-cc{font-family:monospace;font-size:.6rem;color:__ACCENT__;
    white-space:nowrap;}
  .vs-graph{width:100%;height:66px;display:block;}
  .vs-graph line{stroke:#d9d2ca;stroke-width:1;stroke-dasharray:60;
    stroke-dashoffset:60;transition:stroke-dashoffset .6s;}
  .in .vs-graph line{stroke-dashoffset:0;}
  .vs-graph circle{fill:__ACCENT__;opacity:0;transform:scale(0);
    transform-origin:center;transition:.4s;}
  .in .vs-graph circle{opacity:1;transform:scale(1);}
  .vs-cf{font-size:.58rem;color:#a49d95;letter-spacing:.06em;margin-top:4px;
    text-align:center;}

  /* search chapter (reuses .vs-grid / .vs-card) */
  .se-query{max-width:640px;margin:2px auto 16px;background:#fff;
    border:1px solid #e4dfda;border-radius:8px;padding:11px 14px;font-size:.9rem;
    display:flex;align-items:center;gap:10px;justify-content:center;
    opacity:0;transform:translateY(-8px);transition:.5s;}
  .in .se-query{opacity:1;transform:none;}
  .se-qvec{display:inline-flex;gap:2px;}
  .se-qvec .cell{width:6px;height:14px;border-radius:1px;}
  #seGrid{max-width:820px;margin:0 auto;}
  /* routed / dimmed index states */
  .vs-card.dim{opacity:.32;filter:grayscale(.7);transition:opacity .5s,filter .5s;}
  .vs-card.active{box-shadow:0 0 0 2px __ACCENT__;border-top-color:__ACCENT__;}
  .vs-card .vs-badge{display:none;font-size:.55rem;font-weight:700;color:#fff;
    background:__ACCENT__;border-radius:3px;padding:1px 5px;margin-left:6px;
    letter-spacing:.04em;}
  .vs-card.active .vs-badge{display:inline-block;}
  /* HNSW walk : entry → visited → hit → neighbours */
  .vs-graph circle.entry{fill:#fff;stroke:#8a6d3b;stroke-width:2.5;r:5;}
  .vs-graph circle.visit{fill:#d95f16;r:4;}
  .vs-graph circle.hit{fill:__ACCENT__;stroke:__ACCENT__;stroke-width:7;
    stroke-opacity:.3;r:5;}
  .vs-graph circle.neighbor{fill:#8a6d3b;r:4.5;stroke:#8a6d3b;stroke-width:5;
    stroke-opacity:.25;}
  .vs-graph line.walk{stroke:__ACCENT__;stroke-width:2;stroke-dashoffset:0;
    opacity:0;transition:opacity .3s;}
  .vs-graph line.walk.on{opacity:1;}
  .vs-graph line.nbr{stroke:#8a6d3b;stroke-width:1.6;stroke-dasharray:3 2;
    stroke-dashoffset:0;opacity:0;transition:opacity .3s;}
  .vs-graph line.nbr.on{opacity:.9;}
  /* search stepper */
  .se-steps{list-style:none;display:flex;gap:8px;max-width:820px;
    margin:0 auto 14px;padding:0;flex-wrap:wrap;}
  .se-steps li{flex:1 1 150px;min-width:140px;border:1px solid #eae5df;
    border-radius:7px;padding:7px 10px;background:#faf8f6;opacity:.4;
    transition:.4s;display:flex;flex-direction:column;gap:2px;position:relative;
    counter-increment:se;}
  .se-steps li::before{content:counter(se);position:absolute;top:-8px;left:9px;
    width:17px;height:17px;border-radius:50%;background:#cfc8c0;color:#fff;
    font-size:.6rem;font-weight:700;display:flex;align-items:center;
    justify-content:center;transition:.4s;}
  .se-steps{counter-reset:se;}
  .se-steps li.on{opacity:1;border-color:__ACCENT__;background:#fff;}
  .se-steps li.on::before{background:__ACCENT__;}
  .se-steps li b{font-size:.72rem;color:#2b2622;}
  .se-steps li span{font-size:.64rem;color:#7a746d;line-height:1.3;}
  .se-results{max-width:640px;margin:18px auto 0;display:flex;
    flex-direction:column;gap:8px;}
  .se-r{border:1px solid #e4dfda;border-left:3px solid __ACCENT__;
    border-radius:6px;padding:8px 12px;background:#fff;font-size:.82rem;
    display:flex;justify-content:space-between;align-items:baseline;gap:10px;
    opacity:0;transform:translateY(8px);transition:.45s;}
  .se-r.on{opacity:1;transform:none;}
  .se-r .se-rsrc{color:#8f8b86;font-size:.66rem;}
  .se-r .sc{font-family:monospace;color:__ACCENT__;font-size:.76rem;}

  /* generation chapter */
  .gen{max-width:640px;margin:6px auto 0;display:flex;flex-direction:column;gap:6px;}
  .gen-lbl{display:flex;align-items:center;gap:8px;font-size:.8rem;font-weight:600;
    color:#2b2622;margin-top:8px;}
  .gen-lbl em{font-style:normal;font-weight:400;font-size:.7rem;color:#8f8b86;}
  .gen-n{flex:0 0 20px;height:20px;border-radius:50%;background:__ACCENT__;color:#fff;
    font-size:.68rem;font-weight:700;display:flex;align-items:center;justify-content:center;}
  .gen-chunks{display:flex;flex-direction:column;gap:7px;}
  .gc{border:1px solid #e4dfda;border-left:3px solid #d9d2ca;border-radius:6px;
    padding:8px 11px;background:#fff;font-size:.8rem;position:relative;
    transition:transform .55s cubic-bezier(.4,0,.2,1),border-color .4s,box-shadow .4s;}
  .gc.top{border-left-color:__ACCENT__;box-shadow:0 2px 10px -6px __ACCENT__;}
  .gc-h{display:flex;justify-content:space-between;align-items:baseline;gap:8px;
    margin-bottom:3px;}
  .gc-lbl{font-weight:600;color:#2b2622;font-size:.74rem;}
  .gc-lbl .rk{display:inline-block;background:__ACCENT__;color:#fff;font-size:.56rem;
    font-weight:700;border-radius:3px;padding:0 5px;margin-right:6px;opacity:0;
    transition:.3s;}
  .gc.reranked .gc-lbl .rk{opacity:1;}
  .gc-sc{font-family:monospace;font-size:.68rem;color:#a49d95;white-space:nowrap;}
  .gc-sc b{color:__ACCENT__;}
  .gc-tx{color:#55504b;line-height:1.4;}
  .gc.copy{animation:gcCopy .5s ease;}
  @keyframes gcCopy{0%{background:#fff;}40%{background:__ACCENT__1c;}100%{background:#fff;}}
  .gen-flow{text-align:center;font-size:.68rem;color:#b0aaa3;letter-spacing:.12em;
    padding:6px 0;opacity:0;transition:.5s;}
  .gen-flow.on{opacity:1;}
  .paste-badge{font-size:.6rem;font-weight:700;color:#fff;background:#8a6d3b;
    border-radius:4px;padding:2px 7px;letter-spacing:.04em;opacity:0;transform:scale(.8);
    transition:.35s;}
  .paste-badge.on{opacity:1;transform:scale(1);}
  .paste-badge.flash{animation:pasteFlash .5s ease;}
  @keyframes pasteFlash{50%{background:__ACCENT__;transform:scale(1.12);}}
  .prompt-card{background:#1f1c19;border-radius:9px;padding:12px 13px;
    font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.72rem;
    color:#e8e2da;display:flex;flex-direction:column;gap:9px;
    opacity:0;transform:translateY(10px);transition:.5s;}
  .prompt-card.on{opacity:1;transform:none;}
  .pc-line{display:flex;gap:9px;align-items:flex-start;line-height:1.45;}
  .pc-tag{flex:0 0 62px;font-size:.56rem;font-weight:700;letter-spacing:.08em;
    color:#9c948a;padding-top:2px;}
  .pc-sys span:last-child{color:#b8b0a6;}
  .pc-q span:last-child{color:#ffd9b0;}
  .pc-ctx-body{display:flex;flex-direction:column;gap:5px;flex:1;}
  .pc-chunk{background:#2b2723;border-left:2px solid __ACCENT__;border-radius:4px;
    padding:4px 8px;color:#e0d8ce;opacity:0;transform:translateX(-8px);
    transition:.4s;}
  .pc-chunk.on{opacity:1;transform:none;}
  .pc-chunk .src{color:#9c948a;font-size:.6rem;}
  .pc-empty{color:#6a635b;font-style:italic;}
  .answer{background:__ACCENT__0f;border:1px dashed __ACCENT__;border-radius:8px;
    padding:12px 14px;font-size:.86rem;color:#3d3a36;line-height:1.5;
    opacity:0;transform:translateY(8px);transition:.5s;}
  .answer.on{opacity:1;transform:none;}
  .answer .cite{color:__ACCENT__;font-weight:600;}
</style>

<script>
(function(){
  var DOC="__DOCSRC__";
  var DOCNAME=__DOCNAME__;   // JSON-encoded — safe with apostrophes / accents
  var root=document.getElementById('rag-story');

  // ── fullscreen ──────────────────────────────────────────────────────────
  var fsBtn=document.getElementById('fsBtn');
  function inFull(){return document.fullscreenElement||document.webkitFullscreenElement;}
  function reqFull(el){var r=el.requestFullscreen||el.webkitRequestFullscreen||
    el.mozRequestFullScreen;if(r)return r.call(el);}
  function exitFull(){var x=document.exitFullscreen||document.webkitExitFullscreen;
    if(x)x.call(document);}
  if(fsBtn){fsBtn.addEventListener('click',function(){
    if(inFull()){exitFull();return;}
    try{var fe=window.frameElement;if(fe){fe.setAttribute('allowfullscreen','');
      fe.setAttribute('allow','fullscreen');}}catch(e){}
    reqFull(root);
  });}
  function syncBtn(){if(fsBtn)fsBtn.textContent=inFull()?'⤢ Quitter':'⛶ Plein écran';}
  document.addEventListener('fullscreenchange',syncBtn);
  document.addEventListener('webkitfullscreenchange',syncBtn);

  function pageInto(el){
    if(DOC){var i=document.createElement('img');i.src=DOC;el.appendChild(i);return i;}
    var f=document.createElement('div');f.className='pdf-fallback';el.appendChild(f);return null;
  }
  pageInto(document.getElementById('pdf1'));

  // ── Slicing scene ───────────────────────────────────────────────────────
  var N=4, OVERLAP=0.16;             // 4 chunks, 16% overlap
  var slicesEl=document.getElementById('slices');
  var TAGS=__TAGS__;
  var TOKS=['~120 tok','~120 tok','~118 tok','~124 tok'];
  var META=TAGS.map(function(t,i){return [['page 1'],[TOKS[i]||'~120 tok'],t];});

  function buildSlices(H){
    slicesEl.innerHTML='';
    var winH=H*(1/(N-(N-1)*OVERLAP));      // window height so windows overlap
    var step=winH*(1-OVERLAP);
    var ovH=winH*OVERLAP;
    for(var i=0;i<N;i++){
      var off=i*step;
      var row=document.createElement('div');row.className='slice-row';
      row.style.transitionDelay=(i*120)+'ms';

      var s=document.createElement('div');s.className='slice';s.style.height=winH+'px';
      // windowed view of the page (the real overlapping region)
      if(DOC){var im=document.createElement('img');im.src=DOC;im.style.marginTop=(-off)+'px';
        im.style.height=H+'px';s.appendChild(im);}
      else{s.style.background='repeating-linear-gradient(#fff,#fff 20px,#f2eee8 20px,#f2eee8 28px)';}
      // overlap bands — the SAME region highlighted on BOTH neighbours:
      // bottom of this chunk == top of the next. No text over the image.
      if(i<N-1){var b=document.createElement('div');b.className='ov-band';
        b.style.height=ovH+'px';b.style.bottom='0';s.appendChild(b);}
      if(i>0){var b2=document.createElement('div');b2.className='ov-band';
        b2.style.height=ovH+'px';b2.style.top='0';s.appendChild(b2);}
      if(i>0){var cl=document.createElement('div');cl.className='cutline';cl.style.top='-1px';
        cl.style.animationDelay=(i*140)+'ms';s.appendChild(cl);}
      row.appendChild(s);

      // metadata BELOW the slice (readable, never clipped)
      var m=document.createElement('div');m.className='meta';
      var chips='<div class="chip">chunk '+(i+1)+'</div><div class="chip">'+DOCNAME+'</div>';
      META[i][0].forEach(function(x){chips+='<div class="chip">'+x+'</div>';});
      META[i][1].forEach(function(x){chips+='<div class="chip">'+x+'</div>';});
      META[i][2].forEach(function(x){chips+='<div class="chip tag">'+x+'</div>';});
      m.innerHTML=chips;row.appendChild(m);

      slicesEl.appendChild(row);
    }
    // legend for the overlap bands (replaces the on-image labels)
    var lg=document.createElement('div');lg.className='ov-legend';
    lg.innerHTML='<span class="sw"></span> zone répétée (overlap) — présente dans les deux chunks';
    slicesEl.appendChild(lg);
  }
  function playSlice(){
    var mid=document.getElementById('ovMid');
    slicesEl.classList.remove('cut','meta-in','slicing');
    if(mid)mid.classList.remove('lit');
    void slicesEl.offsetWidth;
    slicesEl.classList.add('slicing');
    // middle: naive cut, then light the truncated word (the "why")
    setTimeout(function(){if(mid)mid.classList.add('lit');},700);
    // right: slices separate, overlap bands, metadata
    setTimeout(function(){slicesEl.classList.add('cut');},1200);
    setTimeout(function(){slicesEl.classList.add('meta-in');},2000);
  }
  // build the origin page + slices once the image size is known (run ONCE)
  function initSlices(){
    var probe=document.getElementById('pdf2');
    if(probe.hasChildNodes())return;           // already built
    if(!DOC){ pageInto(probe); buildSlices(346); return; }
    var done=false;
    function build(){
      if(done||!measure.naturalWidth)return; done=true;
      var H=260*(measure.naturalHeight/measure.naturalWidth);
      var shown=document.createElement('img');shown.src=DOC;
      probe.appendChild(shown);
      buildSlices(H);
    }
    var measure=new Image();
    measure.onload=build;
    measure.src=DOC;
    if(measure.complete&&measure.naturalWidth)build();   // cached case
  }
  initSlices();
  document.getElementById('replay').addEventListener('click',playSlice);

  // ── middle column: naive cut (a word truncated → the "why overlap") ─────
  var ovMid=document.getElementById('ovMid');
  function ovcard(html,delay,note,noteCls){
    var c=document.createElement('div');c.className='ovcard';
    c.style.transitionDelay=delay+'ms';c.innerHTML=html;
    if(note){var n=document.createElement('div');n.className='ovnote '+noteCls;
      n.textContent=note;c.appendChild(n);}
    return c;
  }
  var NAIVE=__NAIVE__;
  function realCard(label,ctxBefore,cutBefore,cutAfter,ctxAfter,delay,note){
    // built with text nodes → the PDF text can't break the HTML
    var c=document.createElement('div');c.className='ovcard';c.style.transitionDelay=delay+'ms';
    var l=document.createElement('span');l.className='lbl';l.textContent=label;c.appendChild(l);
    if(ctxBefore!=null)c.appendChild(document.createTextNode(ctxBefore));
    if(cutBefore){var w=document.createElement('span');w.className='cut-word';
      w.textContent=cutBefore;c.appendChild(w);}
    if(cutAfter){var w2=document.createElement('span');w2.className='cut-word';
      w2.textContent=cutAfter;c.appendChild(w2);}
    if(ctxAfter!=null)c.appendChild(document.createTextNode(ctxAfter));
    if(note){var n=document.createElement('div');n.className='ovnote bad';
      n.textContent=note;c.appendChild(n);}
    return c;
  }
  if(ovMid){
    if(NAIVE){                                   // real chunk from the uploaded PDF
      ovMid.appendChild(realCard('chunk 1 (réel)','… '+NAIVE.c1,NAIVE.c1_cut,null,null,160));
      ovMid.appendChild(realCard('chunk 2 (réel)',null,null,NAIVE.c2_cut,NAIVE.c2+' …',300,
        '« '+NAIVE.c1_cut+NAIVE.c2_cut+' » coupé en deux — sans overlap, la suite manque.'));
    } else {                                     // crafted example (no upload)
      ovMid.appendChild(ovcard('<span class="lbl">chunk 1</span>… tamponnez sans '+
        'frotter car un <span class="cut-word">sol</span>',160));
      ovMid.appendChild(ovcard('<span class="lbl">chunk 2</span>'+
        '<span class="cut-word">vant</span> abime la fibre et laisse une aureole …',300,
        '« solvant » coupe en deux — sans overlap, la suite manque.','bad'));
    }
  }

  // ── vectors (REAL embeddings from the project's model) ──────────────────
  var EMB=__EMB__;
  function cellColor(v){if(v>=0){var t=v;return'rgb('+Math.round(243*t+250*(1-t))+','+
    Math.round(112*t+247*(1-t))+','+Math.round(33*t+240*(1-t))+')';}var t=-v;
    return'rgb('+Math.round(66*t+250*(1-t))+','+Math.round(133*t+247*(1-t))+','+
    Math.round(244*t+240*(1-t))+')';}
  function heatEl(heat){                       // real floats -> normalized heatmap
    var m=0.0001;heat.forEach(function(x){m=Math.max(m,Math.abs(x));});
    var v=document.createElement('div');v.className='vec';
    heat.forEach(function(x){var c=document.createElement('span');c.className='cell';
      c.style.background=cellColor(Math.max(-1,Math.min(1,x/m)));v.appendChild(c);});
    return v;}
  function floatsEl(preview,ellipsis){         // show it's real floats
    var s=preview.map(function(x){
      var cls=x>0?'pos':(x<0?'neg':'mut');
      return '<span class="'+cls+'">'+(x>=0?'+':'')+x.toFixed(4)+'</span>';
    }).join(', ');
    var d=document.createElement('div');d.className='floats';
    d.innerHTML='[ '+s+(ellipsis?', <span class="mut">…</span>':'')+' ]';return d;}

  var SVGNS='http://www.w3.org/2000/svg';
  function buildSpace(spaceEl,view,refIdx){    // PCA dots + distance lines to ref
    var pts=view.pts,sims=view.sims,r=pts[refIdx];
    var svg=document.createElementNS(SVGNS,'svg');
    for(var i=0;i<pts.length;i++){ if(i===refIdx)continue; var p=pts[i];
      var ln=document.createElementNS(SVGNS,'line');
      ln.setAttribute('x1',r[0]);ln.setAttribute('y1',r[1]);
      ln.setAttribute('x2',p[0]);ln.setAttribute('y2',p[1]);
      ln.setAttribute('stroke','__ACCENT__');
      ln.setAttribute('stroke-width',(0.5+sims[i]*2.4).toFixed(2));
      ln.setAttribute('stroke-opacity',Math.max(0.18,sims[i]).toFixed(2));
      ln.setAttribute('stroke-dasharray','3 3');svg.appendChild(ln);
      var tx=(r[0]+p[0])/2,ty=(r[1]+p[1])/2;
      var t=document.createElementNS(SVGNS,'text');
      t.setAttribute('x',tx);t.setAttribute('y',ty-2);t.setAttribute('font-size','7');
      t.setAttribute('fill','#6a655f');t.setAttribute('text-anchor','middle');
      t.textContent=sims[i].toFixed(2);svg.appendChild(t);
    }
    spaceEl.appendChild(svg);
    for(var i=0;i<pts.length;i++){
      var d=document.createElement('div');
      d.className='pca-dot'+(i===refIdx?' ref':(sims[i]<0.6?' far':''));
      d.style.left=pts[i][0]+'px';d.style.top=pts[i][1]+'px';spaceEl.appendChild(d);
      var nm=document.createElement('div');nm.className='pca-name';
      nm.style.left=pts[i][0]+'px';nm.style.top=(pts[i][1]-11)+'px';
      nm.textContent=(i===refIdx?'◎ question':('c'+(i+1)));spaceEl.appendChild(nm);
    }
  }

  var badge=document.getElementById('modelBadge');
  if(EMB){
    badge.innerHTML='modèle <b>'+EMB.model+'</b> — la dimension du vecteur '+
      '<b>dépend du modèle</b> : ici '+EMB.full.dim+' valeurs (à gauche), '+
      'réduit à '+EMB.small.dim+' (à droite).';
    document.getElementById('hFull').innerHTML='<b>'+EMB.full.dim+'</b> dimensions · '+
      EMB.full.dim+' floats / vecteur';
    document.getElementById('hSmall').innerHTML='<b>'+EMB.small.dim+'</b> dimensions '+
      '(modèle réduit) · seulement '+EMB.small.dim+' floats';
    var eF=document.getElementById('embFull');
    EMB.full.rows.forEach(function(c,i){
      var r=document.createElement('div');r.className='emb';r.style.transitionDelay=(i*160)+'ms';
      var h=document.createElement('div');h.className='emb-head';
      h.innerHTML='<span class="txt">'+c.label+'</span><span class="dim">∈ ℝ^'+EMB.full.dim+'</span>';
      r.appendChild(h);r.appendChild(floatsEl(c.preview,true));r.appendChild(heatEl(c.heat));
      eF.appendChild(r);});
    var eS=document.getElementById('embSmall');
    EMB.small.rows.forEach(function(c,i){
      var r=document.createElement('div');r.className='emb';r.style.transitionDelay=(i*160)+'ms';
      var h=document.createElement('div');h.className='emb-head';
      h.innerHTML='<span class="txt">'+c.label+'</span><span class="dim">∈ ℝ^'+EMB.small.dim+'</span>';
      r.appendChild(h);r.appendChild(floatsEl(c.vals,false));r.appendChild(heatEl(c.heat));
      eS.appendChild(r);});
    buildSpace(document.getElementById('spFull'),EMB.full,EMB.ref);
    buildSpace(document.getElementById('spSmall'),EMB.small,EMB.ref);
  } else {
    badge.textContent='(modèle d\'embedding indisponible — démo hors-ligne)';
  }

  // ── Vector store : index cards + mini HNSW graph + routing ──────────────
  var INDEXES=__INDEXES__;
  var vsGrid=document.getElementById('vsGrid');
  var vsFeed=document.getElementById('vsFeed');
  var vsCards=[];

  function nodesFor(seed){                 // deterministic pseudo-layout
    var pts=[],n=6,a=seed*2.3+1;
    for(var i=0;i<n;i++){
      a=(a*1.7+0.9)%6.2831853;            // cheap deterministic angle walk
      var rad=16+((seed*7+i*13)%18);
      pts.push([60+rad*Math.cos(a+i*1.05),33+rad*Math.sin(a+i*1.05)]);
    }
    return pts;
  }
  function graphSvg(seed){
    var p=nodesFor(seed),edges='',dots='';
    for(var i=0;i<p.length;i++){          // connect each node to its 2 nearest
      var d=[];
      for(var j=0;j<p.length;j++) if(j!==i)
        d.push([Math.hypot(p[i][0]-p[j][0],p[i][1]-p[j][1]),j]);
      d.sort(function(a,b){return a[0]-b[0];});
      for(var k=0;k<2;k++){var j=d[k][1];
        edges+='<line x1="'+p[i][0].toFixed(1)+'" y1="'+p[i][1].toFixed(1)+
               '" x2="'+p[j][0].toFixed(1)+'" y2="'+p[j][1].toFixed(1)+
               '" style="transition-delay:'+(i*40)+'ms"/>';}
    }
    for(var i=0;i<p.length;i++)
      dots+='<circle cx="'+p[i][0].toFixed(1)+'" cy="'+p[i][1].toFixed(1)+
            '" r="'+(i===0?4:3)+'" style="transition-delay:'+(300+i*70)+'ms"/>';
    return '<svg class="vs-graph" viewBox="0 0 120 66">'+edges+dots+'</svg>';
  }
  INDEXES.forEach(function(ix,i){
    var c=document.createElement('div');c.className='vs-card';
    c.style.transitionDelay=(i*90)+'ms';
    c.innerHTML='<div class="vs-ch"><span class="vs-ci">'+ix[0]+
      '</span><span class="vs-cc">'+ix[1].toLocaleString('fr-FR')+' vec.</span></div>'+
      graphSvg(i)+'<div class="vs-cf">index HNSW</div>';
    vsGrid.appendChild(c);vsCards.push(c);
  });

  // routing animation : a vector flies from the feed to each index card
  function routeVectors(){
    if(!vsCards.length) return;
    var fb=vsFeed.getBoundingClientRect();
    vsCards.forEach(function(card,i){
      var t=document.createElement('div');t.className='vs-tok';
      t.style.left=(8+i*10)+'%';vsFeed.appendChild(t);
      var cb=card.getBoundingClientRect();
      t.style.setProperty('--dx',(cb.left+cb.width/2-fb.left-(fb.width*(8+i*10)/100))+'px');
      t.style.setProperty('--dy',(cb.top-fb.top)+'px');
      setTimeout(function(){t.classList.add('fly');
        setTimeout(function(){card.classList.add('pulse');
          setTimeout(function(){card.classList.remove('pulse');},450);},650);
        setTimeout(function(){t.remove();},950);
      },i*220);
    });
  }
  var _vsDone=false;
  function playVs(){routeVectors();}
  var replayVs=document.getElementById('replayVs');
  if(replayVs) replayVs.onclick=playVs;

  // ── Search chapter : query routes to relevant indexes, walks HNSW ────────
  var QIDX=__QIDX__;                        // index positions the query hits
  var seGrid=document.getElementById('seGrid');
  var seCards=[];
  // k nearest neighbours of a node (same rule as the drawn graph edges)
  function nearest(p,i,k){
    var d=[];
    for(var j=0;j<p.length;j++) if(j!==i)
      d.push([Math.hypot(p[i][0]-p[j][0],p[i][1]-p[j][1]),j]);
    d.sort(function(a,b){return a[0]-b[0];});
    return d.slice(0,k).map(function(x){return x[1];});
  }
  // greedy descent : from a fixed entry node, hop to the nearest UNVISITED
  // node until no neighbour is closer — that terminal node is the "hit".
  function walkPath(seed){
    var p=nodesFor(seed),cur=p.length-1,seen={},path=[cur];seen[cur]=1;
    for(var s=0;s<3;s++){
      var nb=nearest(p,cur,3),best=-1,bd=1e9;
      for(var t=0;t<nb.length;t++){var j=nb[t];if(seen[j])continue;
        var dd=Math.hypot(p[cur][0]-p[j][0],p[cur][1]-p[j][1]);
        if(dd<bd){bd=dd;best=j;}}
      if(best<0) break;seen[best]=1;path.push(best);cur=best;
    }
    var hit=path[path.length-1];
    // neighbours of the hit that aren't already on the walked path
    var nbrs=nearest(p,hit,3).filter(function(j){return path.indexOf(j)<0;}).slice(0,2);
    return {pts:p,path:path,hit:hit,nbrs:nbrs};
  }
  if(seGrid) INDEXES.forEach(function(ix,i){
    var c=document.createElement('div');c.className='vs-card';
    c.style.transitionDelay=(i*80)+'ms';
    c.innerHTML='<div class="vs-ch"><span class="vs-ci">'+ix[0]+
      '<span class="vs-badge">routé</span></span>'+
      '<span class="vs-cc">'+ix[1].toLocaleString('fr-FR')+' vec.</span></div>'+
      graphSvg(i)+'<div class="vs-cf">index HNSW</div>';
    seGrid.appendChild(c);seCards.push(c);
  });
  function mkLine(svg,a,b,cls){
    var ln=document.createElementNS('http://www.w3.org/2000/svg','line');
    ln.setAttribute('x1',a[0].toFixed(1));ln.setAttribute('y1',a[1].toFixed(1));
    ln.setAttribute('x2',b[0].toFixed(1));ln.setAttribute('y2',b[1].toFixed(1));
    ln.setAttribute('class',cls);svg.appendChild(ln);return ln;
  }
  // pre-draw hidden walk + neighbour segments inside a card's svg
  function armWalk(card,seed){
    var svg=card.querySelector('svg'),w=walkPath(seed),p=w.pts;
    var seg=[];
    for(var k=0;k<w.path.length-1;k++)
      seg.push(mkLine(svg,p[w.path[k]],p[w.path[k+1]],'walk'));
    var nseg=w.nbrs.map(function(nb){return mkLine(svg,p[w.hit],p[nb],'nbr');});
    return {svg:svg,path:w.path,hit:w.hit,nbrs:w.nbrs,seg:seg,nseg:nseg};
  }
  var seSteps=document.querySelectorAll('#seSteps li');
  function step(n){seSteps.forEach(function(li){
    li.classList.toggle('on',+li.getAttribute('data-s')<=n);});}

  function playSearch(){
    if(!seCards.length) return;
    var qvec=document.getElementById('seQvec');
    var seRes=document.getElementById('seResults');
    if(seRes) seRes.innerHTML='';
    seSteps.forEach(function(li){li.classList.remove('on');});
    seCards.forEach(function(c){c.className='vs-card';
      c.querySelectorAll('circle').forEach(function(ci){
        ci.classList.remove('entry','visit','hit','neighbor');});
      c.querySelectorAll('line.walk,line.nbr').forEach(function(l){l.remove();});});
    // 1) query becomes a vector
    if(qvec){qvec.innerHTML='';if(EMB) qvec.appendChild(heatEl(EMB.query.heat));}
    // 2) route : dim irrelevant indexes, activate the hit ones
    setTimeout(function(){
      seCards.forEach(function(c,i){
        c.classList.add(QIDX.indexOf(i)<0?'dim':'active');});
      // 3) inside each active index : entry → descent → hit → neighbours
      QIDX.forEach(function(idx,order){
        var card=seCards[idx];if(!card) return;
        var w=armWalk(card,idx),circles=card.querySelectorAll('circle');
        var base=order*260;
        // entry point
        setTimeout(function(){circles[w.path[0]].classList.add('entry');
          if(order===0) step(0);},base);
        // greedy descent, one hop at a time
        for(var s=1;s<w.path.length;s++)(function(s){
          setTimeout(function(){
            if(w.seg[s-1]) w.seg[s-1].classList.add('on');
            var isHit=s===w.path.length-1;
            circles[w.path[s]].classList.remove('entry');
            circles[w.path[s]].classList.add(isHit?'hit':'visit');
            if(order===0) step(isHit?2:1);
          },base+300+s*360);
        })(s);
        // neighbours of the hit become the candidate results
        setTimeout(function(){
          w.nseg.forEach(function(l){l.classList.add('on');});
          w.nbrs.forEach(function(nb){circles[nb].classList.add('neighbor');});
          if(order===0) step(3);
        },base+300+w.path.length*360+250);
      });
      // 4) results rise up once the walks are done
      setTimeout(showSeResults,QIDX.length*260+2000);
    },700);
  }
  function showSeResults(){
    var seRes=document.getElementById('seResults');if(!seRes) return;
    seRes.innerHTML='';
    var top=__MATCHES__.slice(0,3);
    top.forEach(function(m,i){
      var hitName=INDEXES[QIDX[i%QIDX.length]]?INDEXES[QIDX[i%QIDX.length]][0]:'';
      var tag=i===0?'vecteur le plus proche':'voisin immédiat';
      var r=document.createElement('div');r.className='se-r';
      r.innerHTML='<span>'+m[0]+'<div class="se-rsrc">'+tag+' · index : '+hitName+
        '</div></span><span class="sc">'+m[1]+'</span>';
      seRes.appendChild(r);
      setTimeout(function(){r.classList.add('on');},120+i*180);
    });
  }
  var _seDone=false;
  var replaySe=document.getElementById('replaySe');
  if(replaySe) replaySe.onclick=playSearch;

  // ── Generation chapter : retrieve → rerank → paste → send ────────────────
  var GEN=__GEN__;
  var genChunks=document.getElementById('genChunks');
  var gcEls=[];
  function short(t,n){return t.length>n?t.slice(0,n).replace(/\s+\S*$/,'')+'…':t;}
  if(genChunks&&GEN){
    // render in retrieval order first
    GEN.retrOrder.forEach(function(ci){
      var c=GEN.chunks[ci];
      var el=document.createElement('div');el.className='gc';el.dataset.ci=ci;
      el.innerHTML='<div class="gc-h"><span class="gc-lbl">'+
        '<span class="rk"></span>'+c.label+'</span>'+
        '<span class="gc-sc" data-sc>récup <b>'+c.retr.toFixed(2)+'</b></span></div>'+
        '<div class="gc-tx">'+short(c.text,110)+'</div>';
      genChunks.appendChild(el);gcEls.push(el);
    });
  }
  function pcChunkHtml(c){
    return '<div class="pc-chunk"><span class="src">['+c.label+'] </span>'+
      short(c.text,150)+'</div>';
  }
  function playGen(){
    if(!genChunks||!GEN) return;
    var flowP=document.getElementById('flowPrompt');
    var flowS=document.getElementById('flowSend');
    var badge=document.getElementById('pasteBadge');
    var card=document.getElementById('promptCard');
    var phase=document.getElementById('genPhase');
    var ans=document.getElementById('genAnswer');
    // reset
    [flowP,flowS,badge,card,ans].forEach(function(x){if(x)x.classList.remove('on');});
    if(ans)ans.innerHTML='';
    document.getElementById('pcSys').textContent=GEN.sys;
    document.getElementById('pcQ').textContent='« '+GEN.query+' »';
    document.getElementById('pcCtx').innerHTML=
      '<span class="pc-empty">— en attente des chunks —</span>';
    gcEls.forEach(function(el){el.className='gc';
      el.querySelector('[data-sc]').innerHTML='récup <b>'+
        GEN.chunks[el.dataset.ci].retr.toFixed(2)+'</b>';});
    if(phase)phase.textContent='— tri par similarité';

    // 1) RERANK : FLIP reorder gcEls into rerankOrder, add rerank scores
    setTimeout(function(){
      if(phase)phase.textContent='— reclassé par le cross-encoder (rerank)';
      var first={};gcEls.forEach(function(el){first[el.dataset.ci]=el.getBoundingClientRect().top;});
      // reorder DOM
      GEN.rerankOrder.forEach(function(ci){
        var el=gcEls.find(function(e){return +e.dataset.ci===ci;});
        genChunks.appendChild(el);
      });
      // FLIP
      gcEls.forEach(function(el){
        var last=el.getBoundingClientRect().top,dy=first[el.dataset.ci]-last;
        el.style.transform='translateY('+dy+'px)';el.style.transition='none';
      });
      requestAnimationFrame(function(){gcEls.forEach(function(el){
        el.style.transition='';el.style.transform='';});});
      // update scores + top highlight
      gcEls.forEach(function(el){var c=GEN.chunks[el.dataset.ci];
        el.classList.add('reranked');
        el.querySelector('.rk').textContent='#'+(c.rank+1);
        el.querySelector('[data-sc]').innerHTML='récup '+c.retr.toFixed(2)+
          ' → rerank <b>'+c.rerank.toFixed(2)+'</b>';
        if(c.rank===0)el.classList.add('top');});
    },800);

    // 2) PASTE : copy each top chunk's text into the prompt CONTEXT
    setTimeout(function(){
      if(flowP)flowP.classList.add('on');
      if(card)card.classList.add('on');
      if(badge)badge.classList.add('on');
      document.getElementById('pcCtx').innerHTML='';
      var ord=GEN.rerankOrder.slice(0,3);
      ord.forEach(function(ci,k){
        var c=GEN.chunks[ci];
        setTimeout(function(){
          // flash the source chunk (Ctrl+C) then insert into prompt (Ctrl+V)
          var srcEl=gcEls.find(function(e){return +e.dataset.ci===ci;});
          if(srcEl)srcEl.classList.add('copy');
          if(badge){badge.classList.add('flash');
            setTimeout(function(){badge.classList.remove('flash');},500);}
          var d=document.createElement('div');d.innerHTML=pcChunkHtml(c);
          var node=d.firstChild;document.getElementById('pcCtx').appendChild(node);
          requestAnimationFrame(function(){node.classList.add('on');});
          if(srcEl)setTimeout(function(){srcEl.classList.remove('copy');},500);
        },k*650);
      });
    },1900);

    // 3) SEND → answer
    setTimeout(function(){
      if(flowS)flowS.classList.add('on');
    },1900+3*650+300);
    setTimeout(function(){
      if(!ans)return;
      var best=GEN.chunks[GEN.rerankOrder[0]];
      ans.innerHTML='D\'après le contexte fourni : '+short(best.text,150)+
        ' <span class="cite">['+best.label+']</span>';
      ans.classList.add('on');
    },1900+3*650+900);
  }
  var _genDone=false;
  var replayGen=document.getElementById('replayGen');
  if(replayGen) replayGen.onclick=playGen;

  // ── scroll reveal ───────────────────────────────────────────────────────
  var chaps=root.querySelectorAll('.chap');var dots=root.querySelectorAll('.rail span');
  var io=new IntersectionObserver(function(es){es.forEach(function(e){
    if(e.isIntersecting){e.target.classList.add('in');
      var i=+e.target.getAttribute('data-i');
      dots.forEach(function(d){d.classList.toggle('on',+d.getAttribute('data-c')===i);});
      if(i===1)setTimeout(playSlice,250);
      if(i===3&&!_vsDone){_vsDone=true;setTimeout(playVs,500);}
      if(i===4&&!_seDone){_seDone=true;setTimeout(playSearch,500);}
      if(i===5&&!_genDone){_genDone=true;setTimeout(playGen,400);}
    }});},{threshold:0.4});
  chaps.forEach(function(c){io.observe(c);});
})();
</script>
"""

st_html(
    _STORY.replace("__ACCENT__", _ACCENT)
          .replace("__QUERY__", _DEMO_QUERY)
          .replace("__TAGS__", _tags_json)
          .replace("__MATCHES__", _matches_json)
          .replace("__INDEXES__", _indexes_json)
          .replace("__QIDX__", _qidx_json)
          .replace("__GEN__", _gen_json)
          .replace("__DOCNAME__", _doc_name_json)
          .replace("__DOCSRC__", doc_src)
          .replace("__EMB__", _emb_json)
          .replace("__NAIVE__", _naive_json),
    height=900, scrolling=True,
)
