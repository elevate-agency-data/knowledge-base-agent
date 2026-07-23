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

_STORY = r"""
<div id="rag-story">
  <button class="fs-btn" id="fsBtn" title="Plein écran">⛶ Plein écran</button>
  <div class="rail"><span data-c="0"></span><span data-c="1"></span><span data-c="2"></span>
    <span data-c="3"></span><span data-c="4"></span></div>

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

  <!-- 3 store -->
  <section class="chap" data-i="3">
    <div class="step">04 — Indexation</div>
    <h2>On stocke les vecteurs</h2>
    <p class="lead">Embeddings + métadonnées entrent dans une base vectorielle
       indexée (HNSW), interrogeable en millisecondes.</p>
    <div class="stage"><div class="db"><div class="db-top"></div>
      <div class="db-body"><div class="db-rows" id="dbRows"></div></div>
      <div class="db-bot"></div><div class="db-cap">vector store · HNSW</div></div></div>
  </section>

  <!-- 4 retrieval -->
  <section class="chap" data-i="4">
    <div class="step">05 — Retrieval + génération</div>
    <h2>Une question retrouve les bons chunks</h2>
    <p class="lead">La question devient un vecteur, on cherche les plus proches,
       le LLM répond <b>sur ces sources</b>.</p>
    <div class="stage retr">
      <div class="q">« __QUERY__ » <span class="qv" id="qv"></span></div>
      <div class="matches" id="matches"></div>
      <div class="answer">Réponse sourcée, citations à l'appui.</div>
    </div>
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

  /* db */
  .db{width:200px;position:relative;opacity:0;transform:translateY(16px);transition:.6s;}
  .in .db{opacity:1;transform:none;}
  .db-top{height:22px;border-radius:50%;background:__ACCENT__;}
  .db-body{background:linear-gradient(90deg,__ACCENT__,#d95f16);height:120px;margin-top:-11px;overflow:hidden;}
  .db-bot{height:22px;border-radius:50%;background:#c9560f;margin-top:-11px;}
  .db-cap{text-align:center;font-size:.62rem;color:#8f8b86;margin-top:10px;letter-spacing:.08em;}
  .db-rows{padding:16px 14px;display:flex;flex-direction:column;gap:7px;}
  .db-row{height:8px;background:#ffffffbb;border-radius:2px;width:0;transition:width .5s;}
  .in .db-row{width:100%;}

  /* retrieval */
  .retr{flex-direction:column;align-items:stretch;gap:14px;max-width:520px;margin:26px auto 0;}
  .q{background:#fff;border:1px solid #e4dfda;border-radius:8px;padding:12px 14px;font-size:.9rem;
     display:flex;align-items:center;gap:10px;}
  .qv{display:inline-flex;gap:2px;}.qv .cell{width:6px;height:14px;}
  .matches{display:flex;flex-direction:column;gap:8px;}
  .m{border:1px solid #e4dfda;border-left:3px solid __ACCENT__;border-radius:6px;padding:9px 12px;
     background:#fff;font-size:.82rem;display:flex;justify-content:space-between;
     opacity:0;transform:translateX(-10px);transition:.5s;}
  .in .m{opacity:1;transform:none;}
  .m .sc{font-family:monospace;color:__ACCENT__;font-size:.78rem;}
  .answer{background:__ACCENT__0f;border:1px dashed __ACCENT__;border-radius:8px;padding:12px 14px;
     font-size:.86rem;color:#3d3a36;opacity:0;transition:.6s .3s;}
  .in .answer{opacity:1;}
</style>

<script>
(function(){
  var DOC="__DOCSRC__";
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
      var chips='<div class="chip">chunk '+(i+1)+'</div><div class="chip">__DOCNAME__</div>';
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

  var db=document.getElementById('dbRows');
  for(var i=0;i<7;i++){var r=document.createElement('div');r.className='db-row';
    r.style.transitionDelay=(i*90)+'ms';db.appendChild(r);}

  var qv=document.getElementById('qv');
  if(qv&&EMB){qv.appendChild(heatEl(EMB.query.heat));}
  var matches=document.getElementById('matches');
  __MATCHES__.forEach(function(m,i){
    var d=document.createElement('div');d.className='m';d.style.transitionDelay=(i*160)+'ms';
    d.innerHTML='<span>'+m[0]+'</span><span class="sc">'+m[1]+'</span>';matches.appendChild(d);});

  // ── scroll reveal ───────────────────────────────────────────────────────
  var chaps=root.querySelectorAll('.chap');var dots=root.querySelectorAll('.rail span');
  var io=new IntersectionObserver(function(es){es.forEach(function(e){
    if(e.isIntersecting){e.target.classList.add('in');
      var i=+e.target.getAttribute('data-i');
      dots.forEach(function(d){d.classList.toggle('on',+d.getAttribute('data-c')===i);});
      if(i===1)setTimeout(playSlice,250);
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
          .replace("__DOCNAME__", _doc_name)
          .replace("__DOCSRC__", doc_src)
          .replace("__EMB__", _emb_json)
          .replace("__NAIVE__", _naive_json),
    height=900, scrolling=True,
)
