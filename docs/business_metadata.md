# Business metadata (tag convention)

Business-level metadata rides on the existing `tags VARCHAR[]` column of the
chunk schema — no schema migration, works on every brand's store, and is
already filterable (`build_filters(tags=[...])` → `apply_post_filter`, AND
semantics: a chunk must carry **all** listed tags).

## Convention

Prefix each tag with its facet:

| Facet | Prefix | Example values (hermes SAV) |
|-------|--------|-----------------------------|
| Product type | `produit:` | `produit:sac`, `produit:carre`, `produit:montre` |
| Material | `matiere:` | `matiere:cuir`, `matiere:soie`, `matiere:metal` |
| Request category | `demande:` | `demande:reparation`, `demande:entretien`, `demande:garantie` |
| Audience | `cible:` | `cible:conseiller`, `cible:artisan`, `cible:responsable` |

Tags are lowercase. Add as many as apply to a chunk.

## Where tags are set

At ingestion. Options, cheapest first:

1. **Folder / file mapping** — derive from the L1 domain and file name
   (e.g. everything under `Entretien-Soin/` → `demande:entretien`).
2. **LLM tagging** — a light pass over each chunk to assign `produit:` /
   `matiere:` from the content.

Until the ingestion populates them, the RAG Demo tag filter simply returns
nothing — no error, no fake data.

## Demo usage

RAG Demo page → sidebar → *Filter by tags*:

```
matiere:cuir, demande:entretien
```

Only chunks carrying both tags survive, before dense/sparse/hybrid fusion —
this is the "answer improved by metadata" step of the workshop.
