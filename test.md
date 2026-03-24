# Plan de test — Knowledge Base Agent

URL de test : `https://drive.google.com/file/d/1q0PhLiMtbueKhLLZc2A1Bw7XdkfQDRAX/view?usp=sharing`

---

## Règles de routing agent (référence)

| Situation | Pipeline attendu |
|---|---|
| Client/index mentionné (celio, fnac…) | Hybrid mono-index |
| Plusieurs clients → comparaison | Hybrid multi-index |
| "tous les index" / "tous les clients" | Hybrid `index_names=[]` (tous les index locaux) |
| Aucun client ni index précisé | Vertex (corpus global) |
| URL Drive, pipeline non précisé | `hybrid_find_similar` (index_names=[]) |
| URL Drive + "vertex" / corpus précisé | `vertex_find_similar` |
| URL Drive + "les deux" | `hybrid_find_similar` ET `vertex_find_similar` |
| URL Drive dans `hybrid_query` ou `rag_query` | ❌ Interdit |

---

## Page 1 — Agent Chat

### Vertex AI RAG — gestion

- [X] `liste mes corpus`
- [X] `interroge le corpus base_rag : quels sont les services proposés à Celio ?`
- [X] `lis ce document et dis-moi de quoi il parle : https://drive.google.com/file/d/1q0PhLiMtbueKhLLZc2A1Bw7XdkfQDRAX/view?usp=sharing`

### Vertex AI RAG — routing automatique

- [X] `quelles sont les technologies recommandées dans les propositions Elevate ?`
  - Attendu : `rag_query` — aucun client précisé → Vertex
- [X] `qu'est-ce qu'Elevate propose comme offres ?`
  - Attendu : `rag_query` — question générale → Vertex

### Hybrid RAG — gestion

- [X] `liste les index hybrid`
- [X] `donne-moi le détail de l'index celio`
- [X] `liste le contenu du drive`

### Hybrid RAG — requête texte

- [X] `quels sont les livrables proposés à Celio ?`
  - Attendu : `hybrid_query(index_names=["celio"])`
- [X] `compare les offres Celio et Invivo`
  - Attendu : `hybrid_query(index_names=["celio", "invivo"])`
- [X] `quel est le budget de la prestation pour Aldi ?` *(résultats à revoir : euros ou j/h pas clair)*
  - Attendu : `hybrid_query(index_names=["aldi"])`
- [X] `quels sont les services proposés dans tous les index ?`
  - Attendu : `hybrid_query(index_names=[])` — "tous les index" → tous les index locaux

### Similarité documentaire — Hybrid (défaut, pas de pipeline précisé)

- [X] `Donne moi les fichiers similaires à ce document : https://drive.google.com/file/d/1q0PhLiMtbueKhLLZc2A1Bw7XdkfQDRAX/view?usp=sharing`
  - Attendu : `hybrid_find_similar(index_names=[])` — tous les index automatiquement

### Similarité documentaire — Vertex explicite

- [X] `En utilisant Vertex, trouve les documents similaires à ce fichier : https://drive.google.com/file/d/1q0PhLiMtbueKhLLZc2A1Bw7XdkfQDRAX/view?usp=sharing`
  - Attendu : `vertex_find_similar`

### Similarité documentaire — les deux pipelines

- [X] `Compare les résultats Vertex et Hybrid pour les documents similaires à : https://drive.google.com/file/d/1q0PhLiMtbueKhLLZc2A1Bw7XdkfQDRAX/view?usp=sharing`
  - Attendu : `hybrid_find_similar` ET `vertex_find_similar`

### Cas d'erreur — Index inexistant

- [X] `interroge l'index toto : quels sont les livrables ?`
  - Attendu : message d'erreur clair `"Index inconnu : toto. Index disponibles : celio, fnac, …"` — pas de résultat vide silencieux

### Cas d'erreur — URL ne doit jamais aller dans hybrid_query

- [X] `Recherche dans l'index celio les documents proches de : https://drive.google.com/file/d/1q0PhLiMtbueKhLLZc2A1Bw7XdkfQDRAX/view?usp=sharing`
  - Attendu : `hybrid_find_similar(index_names=[])` — jamais `hybrid_query` avec l'URL

### Continuité conversationnelle

- [X] Envoyer `quels sont les livrables pour Celio ?` puis `et le budget ?`
  - Attendu : deuxième appel avec contexte `context="Q: livrables Celio"`

---

## Page 2 — RAG Comparison

### Client précis — mono-index

- [X] `quels sont les livrables proposés à Celio ?`
  - Attendu : Vertex `rag_query` + Hybrid `hybrid_rag_query(index_name="celio")`

### Comparaison — multi-index

- [X] `compare les offres Celio et Fnac`
  - Attendu : Vertex `rag_query` + Hybrid `hybrid_multi_query(index_names=["celio","fnac"])`

### Question générale — aucun client

- [X] `quelles sont les technologies recommandées dans les propositions Elevate ?`
  - Attendu : Vertex `rag_query` + Hybrid skippé ou tous les index

### URL Drive — find similar (les deux côtés)

- [X] `Donne moi les fichiers similaires à ce document : https://drive.google.com/file/d/1q0PhLiMtbueKhLLZc2A1Bw7XdkfQDRAX/view?usp=sharing`
  - Attendu : `vertex_find_similar` (gauche) + `hybrid_find_similar` (droite)

### Continuité conversationnelle

- [X] `quels sont les livrables pour Celio ?` puis `et pour Fnac c'est quoi ?`
  - Attendu : deuxième requête avec contexte conversationnel dans les deux pipelines

---

## Page 3 — Index Manager

### Tab "Index existants"

- [ ] La liste s'affiche avec chunks / fichiers / modèle
- [ ] Bouton "Actualiser" fonctionne
- [ ] Ouvrir un expander — métriques correctes

### Tab "Créer un index"

- [ ] Créer un index `test-tmp` avec `mpnet-768` / `fixed`
- [ ] Vérifier qu'il apparaît dans la liste
- [ ] Supprimer `test-tmp` — tester la confirmation avant suppression

### Tab "Ingérer des données"

- [ ] Sélectionner un index existant
- [ ] Indiquer un dossier Drive valide
- [ ] Vérifier le retour `files_processed` / `chunks_added`

### Tab "Explorer un dossier Drive"

- [ ] Saisir le nom du dossier racine Drive
- [ ] Vérifier que sous-dossiers et fichiers s'affichent avec liens

---

## Page 4 — Benchmark

> Nécessite que `hybrid/data/benchmark_results.json` existe.

- [ ] Podium top 3 s'affiche
- [ ] Changer la métrique principale : nDCG → MRR → Recall → Precision
- [ ] Filtrer par modèle d'embedding
- [ ] Filtrer par stratégie de chunking
- [ ] Filtrer par mode de retrieval
- [ ] Heatmap embedding × chunking s'affiche correctement

---

## Page 5 — Simple Chat

### Sans RAG (toggle OFF)

- [X] `qu'est-ce qu'une proposition commerciale ?`
  - Attendu : réponse Gemini direct, aucune source

### Hybrid RAG — client précis

> Toggle ON, pipeline `Hybrid RAG`

- [ ] `quels sont les livrables pour Celio ?`
- [ ] `et le budget ?` *(continuité conversationnelle — doit résoudre "Celio" depuis le contexte)*

### Hybrid RAG — URL Drive

> Toggle ON, pipeline `Hybrid RAG`

- [X] `Donne moi les fichiers similaires à ce document : https://drive.google.com/file/d/1q0PhLiMtbueKhLLZc2A1Bw7XdkfQDRAX/view?usp=sharing`
  - Attendu : liste avec score %, index, lien

### Vertex AI — client précis

> Toggle ON, pipeline `Vertex AI`, sélectionner un corpus

- [X] `quels sont les livrables pour Celio ?`

### Vertex AI — question générale

> Toggle ON, pipeline `Vertex AI`

- [X] `quelles sont les technologies recommandées dans les propositions Elevate ?`
  - Attendu : `rag_query` sur le corpus sélectionné

### Vertex AI — URL Drive

> Toggle ON, pipeline `Vertex AI`

- [X] `Donne moi les fichiers similaires à ce document : https://drive.google.com/file/d/1q0PhLiMtbueKhLLZc2A1Bw7XdkfQDRAX/view?usp=sharing`
  - Attendu : liste avec titre et lien Drive
