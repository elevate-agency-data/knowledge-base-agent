# Knowledge Base Agent

Agent de gestion de bases de connaissances documentaires avec deux pipelines RAG complémentaires : **Vertex AI RAG** (cloud GCP) et **Hybrid RAG** (local DuckDB). Piloté via un agent conversationnel Google ADK et une interface Streamlit.

---

## Architecture

```
knowledge-base-agent/
├── rag_agent/          # Pipeline Vertex AI RAG (Google Cloud)
│   ├── agent.py        # Agent ADK principal (gemini-2.5-pro)
│   ├── config.py       # Paramètres GCP, chunking, retrieval
│   ├── simple_chat_agent.py  # Agent chat simplifié (sans outils)
│   └── tools/          # Outils ADK : create_corpus, add_data, rag_query...
│
├── hybrid/             # Pipeline Hybrid RAG (local)
│   ├── config.py       # Paramètres env, embeddings, retrieval, DRIVE_ROOT_FOLDER
│   ├── embeddings/     # 5 modèles : minilm-384, mpnet-768, e5-large-1024, bge-m3, vertex
│   ├── ingestion/      # Extraction Drive (PDF, Docx, Xlsx, Pptx), chunking HuggingFace
│   ├── retrieval/      # Dense (HNSW/cosinus), Sparse (BM25), Hybrid (RRF)
│   ├── stores/         # DuckDB (FLOAT[768] + HNSW VSS) et AlloyDB (GCP)
│   ├── tools/          # Outils ADK : hybrid_query, hybrid_list_drive...
│   └── benchmark/      # Évaluation 90 combinaisons (MRR, nDCG, Recall, Precision)
│
├── shared/
│   ├── query_rewriter.py   # Réécriture sémantique des queries (gemini-2.0-flash)
│   └── index_resolver.py   # Résolution automatique des index via Gemini Flash
│
├── ui/                 # Interface Streamlit (5 pages)
│   ├── app.py
│   ├── pages/
│   │   ├── 1_Agent_Chat.py       # Chat avec l'agent ADK
│   │   ├── 2_RAG_Comparison.py   # Comparaison Vertex vs Hybrid en parallèle
│   │   ├── 3_Index_Manager.py    # Gestion des index Hybrid
│   │   ├── 4_Benchmark.py        # Visualisation résultats benchmark
│   │   └── 5_Simple_Chat.py      # Chat simplifié (toggle RAG on/off)
│   ├── services/
│   │   ├── vertex_service.py     # Wrapper Vertex AI (query, direct_query, synthesize)
│   │   ├── hybrid_service.py     # Wrapper Hybrid RAG (query, resolve_indexes)
│   │   └── session_store.py      # Sessions de comparaison (DuckDB)
│   └── components/     # Composants réutilisables (source_card, chat_message...)
│
├── docs/
│   └── embedding_models.md  # Documentation modèles, chunking, métriques
│
├── run_ui.py           # Lanceur Streamlit
└── requirements.txt
```

---

## Prérequis

- Python 3.12+
- Un projet Google Cloud avec les APIs activées : Vertex AI, Google Drive, IAM
- Un compte de service avec les rôles : `Vertex AI User`, `Storage Object Viewer`
- Google Drive partagé avec le compte de service

---

## Installation

```bash
# Cloner le dépôt
git clone <repo-url>
cd knowledge-base-agent

# Créer et activer l'environnement virtuel
python -m venv rag_venv
source rag_venv/bin/activate       # Linux/Mac
rag_venv\Scripts\activate          # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## Configuration

### 1. Clé de compte de service GCP

Placer le fichier JSON de clé dans `rag_agent/key.json` (ignoré par `.gitignore`).

### 2. Variables d'environnement

Créer un fichier `rag_agent/.env` :

```env
GOOGLE_APPLICATION_CREDENTIALS=rag_agent/key.json
GOOGLE_CLOUD_PROJECT=knowledge-base-agent-485813
GOOGLE_CLOUD_LOCATION=europe-west1
```

### 3. Paramètres Vertex AI (`rag_agent/config.py`)

```python
PROJECT_ID              = "knowledge-base-agent-485813"
LOCATION                = "europe-west1"
MODEL                   = "gemini-2.5-pro"
DEFAULT_TOP_K           = 10
DEFAULT_DISTANCE_THRESHOLD = 0.5
DEFAULT_EMBEDDING_MODEL = "publishers/google/models/text-embedding-005"
```

### 4. Paramètres Hybrid RAG (`hybrid/config.py`)

```python
ENV                     = "local"          # "local" → DuckDB | "gcp" → AlloyDB pour prod
DUCKDB_PATH             = "hybrid/data/hybrid.duckdb"
DRIVE_ROOT_FOLDER       = "Insight Factory - RAG"  # dossier racine Drive
DEFAULT_EMBEDDING_MODEL = "mpnet-768"
CHUNK_SIZE              = 384              # en tokens (= max_seq_length du modèle)
CHUNK_OVERLAP           = 64              # en tokens
DENSE_WEIGHT            = 0.7
SPARSE_WEIGHT           = 0.3
```

`DRIVE_ROOT_FOLDER` est le point d'entrée de toutes les recherches Drive. La recherche de dossiers clients est toujours restreinte à ses enfants directs, évitant les collisions de noms entre clients.

---

## Lancer l'interface Streamlit

```bash
python run_ui.py
# ou directement :
streamlit run ui/app.py
```

L'interface s'ouvre sur `http://localhost:8501`.

---

## Lancer l'agent ADK (terminal)

```bash
cd knowledge-base-agent
adk run rag_agent
# ou en mode web :
adk web
```

---

## Les deux pipelines RAG

### Pipeline 1 — Vertex AI RAG

Corpus hébergés dans Google Cloud. Ingestion et retrieval gérés par l'API Vertex AI RAG.

| Outil ADK | Description |
|---|---|
| `create_corpus` | Crée un nouveau corpus Vertex AI RAG |
| `add_data` | Ingère un dossier Google Drive entier (récursif) |
| `rag_query` | Interroge un corpus, retourne réponse + sources |
| `list_corpora` | Liste tous les corpus disponibles |
| `delete_corpus` | Supprime un corpus (confirmation requise) |
| `get_document_content` | Lit le contenu d'un document via son lien Drive |

**Points forts :** Ingestion massive, pas de ressources locales requises, corpus partagés multi-utilisateurs.

**Limites :** Latence cloud (~14s cold start), coût par requête, pas de recherche document-à-document.

---

### Pipeline 2 — Hybrid RAG

Index locaux par dossier, stockés dans DuckDB (schema `FLOAT[768]` + HNSW index via VSS). Recherche dense + sparse fusionnée par RRF.

| Outil ADK | Description |
|---|---|
| `hybrid_create_index` | Crée un index pour un domaine (ex: "rh") |
| `hybrid_add_data` | Ingère des dossiers Drive dans un index (récursif, tous types de fichiers) |
| `hybrid_query` | Interroge un ou plusieurs index (routing auto single/multi) |
| `hybrid_find_similar` | Trouve les documents similaires à un lien Drive (vecteur à vecteur) |
| `hybrid_list_indexes` | Liste tous les index disponibles |
| `hybrid_index_info` | Détail d'un index (fichiers, chunks, modèle) |
| `hybrid_delete_index` | Supprime un index (confirmation requise) |
| `hybrid_list_drive` | Liste le contenu complet d'un dossier Drive (récursif, tous niveaux) |

**Points forts :** Isolation par dossier, warm start rapide (~0.05s HNSW), recherche multi-index, similarité documentaire.

**Limites :** Ressources locales (RAM selon le modèle d'embedding), pas de partage cloud natif.

#### Types de fichiers supportés à l'ingestion

| Format | Extension | Extraction |
|---|---|---|
| Google Doc | — | Export texte via Drive API |
| Google Sheet | — | Export CSV via Drive API |
| Google Slides | — | Export texte via Drive API |
| PDF | `.pdf` | pymupdf (in-memory) |
| Word | `.docx`, `.doc` | python-docx |
| Excel | `.xlsx`, `.xls` | openpyxl |
| PowerPoint | `.pptx`, `.ppt` | python-pptx |

#### Traversée Drive

L'ingestion et le listing parcourent l'**arborescence complète** (récursif, pagination Drive incluse). La recherche de dossiers est toujours scopée au `DRIVE_ROOT_FOLDER` pour éviter les collisions de noms entre clients.

---

## Résolution automatique des index

Avant chaque requête multi-index, `shared/index_resolver.py` détermine via Gemini Flash quels index interroger en fonction de la query :

- Si la query mentionne un client précis → index correspondant uniquement
- Si la query est générale ou comparative → tous les index

Utilisé par l'outil `hybrid_query`, la page Simple Chat et la page Comparaison RAG.

---

## Réécriture de query (Query Rewriter)

Avant chaque retrieval, la query utilisateur est réécrite par `gemini-2.0-flash` pour améliorer la pertinence sémantique.

```
Query brute : "compare rh et marketing"
Query réécrite : "RH Marketing politiques procédures différences"
```

Le rewriter supprime les verbes d'action et garde les concepts. **Les noms propres (domaines, thématiques, projets) sont toujours conservés** pour éviter de perdre les entités dans l'embedding. Il accepte un paramètre `context` pour résoudre les références conversationnelles. Chaque pipeline (Vertex et Hybrid) reçoit un contexte issu **uniquement de ses propres réponses précédentes** — les contextes ne sont jamais mélangés.

### Grounding Elevate

Toutes les réponses LLM (agent, Simple Chat, Comparaison) sont cadrées dans le contexte Elevate : les documents sont des propositions commerciales et analyses internes. Le LLM ne complète **jamais** avec sa connaissance générale des entreprises.

### Similarité documentaire (Drive URL)

Quand la query contient une URL Google Drive :
- **Hybrid** : `hybrid_find_similar` extrait le contenu du document, encode en vecteur moyen et cherche par distance cosinus — pas de query texte.
- **Vertex** : `extract_query_from_drive_url` extrait le contenu, le résume via Gemini Flash, puis utilise ce résumé comme query de recherche dans le corpus.

---

## Modèles d'embedding disponibles (Hybrid)

| Modèle | Dimension | RAM | Qualité FR/EN | Sparse natif |
|---|---|---|---|---|
| `minilm-384` | 384 | ~200 MB | Bonne | Non |
| `mpnet-768` | 768 | ~600 MB | Tres bonne | Non |
| `e5-large-1024` | 1024 | ~1.5 GB | Tres bonne | Non |
| `bge-m3` | 1024 | ~2.5 GB | Excellente | Oui |
| `vertex` | 768 | 0 (cloud) | Excellente | Non |

Voir `docs/embedding_models.md` pour la documentation complète.

---

## Modes de retrieval (Hybrid)

| Mode | Description | Usage |
|---|---|---|
| `dense` | `array_cosine_similarity` + HNSW index (DuckDB VSS) | Questions sémantiques, paraphrases |
| `sparse` | BM25 via DuckDB FTS | Codes produits, noms propres, jargon exact |
| `hybrid` | Fusion RRF dense + sparse (défaut) | Production générale |

Paramètres de fusion dans `hybrid/config.py` : `DENSE_WEIGHT=0.7`, `SPARSE_WEIGHT=0.3`, `RRF_K=60`.

### Performance dense search (HNSW)

| Méthode | Latence (warm, 10k chunks) |
|---|---|
| NumPy row-by-row (ancien) | ~3.5s |
| `list_cosine_similarity` SQL | ~0.7s |
| `array_cosine_similarity` + HNSW | **~0.05s** |

Le schema `FLOAT[768]` (fixed-size array) est requis par HNSW. L'index est reconstruit automatiquement après chaque ingestion/suppression. Fallback en cascade : `array_cosine_similarity` → `list_cosine_similarity` → NumPy.

---

## Chunking

Le chunking est réalisé en **tokens réels** via le tokenizer HuggingFace du modèle d'embedding (mpnet-768 : `max_seq_length = 384`). Chaque chunk utilise 100% de la fenêtre du modèle — aucune approximation en caractères.

| Stratégie | Description | Usage |
|---|---|---|
| `fixed` (défaut) | 384 tokens, overlap 64, tokenizer HuggingFace | Production générale |
| `semantic` | Coupure aux ruptures sémantiques (seuil cosinus 0.85) | CR de réunion, textes narratifs |
| `hierarchical` | Chunks enfants 256 + parents 1024 | Documents longs multi-niveaux |

---

## Benchmark

Évalue **90 combinaisons** : 5 embeddings × 6 stratégies de chunking × 3 modes de retrieval.

```bash
# Dataset synthétique intégré (aucun prérequis)
python -m hybrid.benchmark.eval_retrieval

# Mode Q&A avec tes vrais documents
python -m hybrid.benchmark.eval_retrieval --mode qa --qa-file mon_dataset.csv

# Restreindre aux modèles légers (éviter OOM)
python -m hybrid.benchmark.eval_retrieval --models minilm-384 mpnet-768

# Options
--output PATH          # fichier de sortie (défaut: hybrid/data/benchmark_results.json)
--test-file PATH       # dataset synthétique personnalisé (JSON)
--qa-file PATH         # fichier Q&A (CSV/Excel : colonnes Q, A, url)
--models [...]         # restreindre les modèles testés
--answer-threshold N   # seuil de chevauchement pour la pertinence (défaut: 3 mots)
```

**Format du fichier Q&A :**

| Q | A | url |
|---|---|---|
| Quelle est la politique de retour ? | Les retours sont acceptés sous 30 jours... | https://docs.google.com/... |

**Métriques calculées :**

| Métrique | Description |
|---|---|
| MRR | Le bon chunk est-il en premier ? |
| nDCG@10 | Les bons chunks sont-ils bien classés ? (métrique principale) |
| Recall@10 | Tous les bons chunks sont-ils retrouvés ? |
| Precision@10 | Parmi les 10 résultats, combien sont pertinents ? |

Les résultats sont visualisables dans la page **Benchmark** de l'UI Streamlit (`4_Benchmark.py`).

---

## Pages de l'interface Streamlit

| Page | Description |
|---|---|
| **Agent Chat** | Chat conversationnel avec l'agent ADK (gemini-2.5-pro), avec affichage des appels d'outils |
| **Comparaison RAG** | Vertex AI vs Hybrid en parallèle sur la même question — sessions persistées en DuckDB, contextes cloisonnés par pipeline, ordre chronologique |
| **Index Manager** | Création, alimentation et suppression des index Hybrid |
| **Benchmark** | Visualisation des résultats benchmark (podium, leaderboard, graphiques, heatmap) |
| **Simple Chat** | Chat simplifié pour utilisateurs non-techniques — toggle RAG on/off, choix Vertex AI ou Hybrid |

---

## Structure des fichiers de données

```
hybrid/data/
├── hybrid.duckdb              # Index Hybrid RAG (ignoré par git)
└── benchmark_results.json     # Résultats du dernier benchmark

ui/data/                       # Données UI (ignorées par git)
├── comparaison/
│   └── comparaison.duckdb     # Sessions de comparaison RAG (DuckDB)
└── agent_display/
    └── {session_id}.json      # Affichage enrichi des sessions agent
```

---

## Dépendances principales

```
# GCP / Agent
google-adk
google-cloud-aiplatform
google-api-python-client

# Hybrid RAG
duckdb
sentence-transformers
transformers           # Tokenizer HuggingFace (chunking en tokens)
FlagEmbedding          # BGE-M3
pymupdf                # Extraction PDF
scikit-learn

# Extraction fichiers Office
python-docx            # .docx / .doc
openpyxl               # .xlsx / .xls
python-pptx            # .pptx / .ppt

# UI
streamlit>=1.35
```
