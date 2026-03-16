# Knowledge Base Agent

Agent de gestion de bases de connaissances documentaires avec deux pipelines RAG complémentaires : **Vertex AI RAG** (cloud GCP) et **Hybrid RAG** (local DuckDB). Piloté via un agent conversationnel Google ADK et une interface Streamlit.

---

## Architecture

```
knowledge-base-agent/
├── rag_agent/          # Pipeline Vertex AI RAG (Google Cloud)
│   ├── agent.py        # Agent ADK principal (gemini-2.5-pro)
│   ├── config.py       # Paramètres GCP, chunking, retrieval
│   └── tools/          # Outils ADK : create_corpus, add_data, rag_query...
│
├── hybrid/             # Pipeline Hybrid RAG (local)
│   ├── config.py       # Paramètres env, embeddings, retrieval
│   ├── embeddings/     # 5 modèles : minilm-384, mpnet-768, e5-large-1024, bge-m3, vertex
│   ├── ingestion/      # Extraction Drive, chunking (fixed/semantic/hierarchical)
│   ├── retrieval/      # Dense (cosinus), Sparse (BM25), Hybrid (RRF)
│   ├── stores/         # DuckDB (local) et AlloyDB (GCP)
│   ├── tools/          # Outils ADK : hybrid_query, hybrid_find_similar...
│   └── benchmark/      # Évaluation 90 combinaisons (MRR, nDCG, Recall, Precision)
│
├── shared/
│   └── query_rewriter.py  # Réécriture sémantique des queries (gemini-2.0-flash)
│
├── ui/                 # Interface Streamlit (4 pages)
│   ├── app.py
│   ├── pages/
│   │   ├── 1_Agent_Chat.py       # Chat avec l'agent ADK
│   │   ├── 2_RAG_Comparison.py   # Comparaison Vertex vs Hybrid en parallèle
│   │   ├── 3_Index_Manager.py    # Gestion des index Hybrid
│   │   └── 4_Benchmark.py        # Visualisation résultats benchmark
│   ├── services/       # Wrappers service pour vertex_service et hybrid_service
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
ENV                     = "local"          # "local" → DuckDB | "gcp" → AlloyDB
DUCKDB_PATH             = "hybrid/data/hybrid.duckdb"
DEFAULT_EMBEDDING_MODEL = "mpnet-768"
CHUNK_SIZE              = 512
DENSE_WEIGHT            = 0.7
SPARSE_WEIGHT           = 0.3
```

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

Index locaux par client, stockés dans DuckDB. Recherche dense + sparse fusionnée par RRF.

| Outil ADK | Description |
|---|---|
| `hybrid_create_index` | Crée un index pour un client (ex: "celio") |
| `hybrid_add_data` | Ingère des dossiers Drive dans un index |
| `hybrid_query` | Interroge un ou plusieurs index (routing auto single/multi) |
| `hybrid_find_similar` | Trouve les documents similaires à un lien Drive (vecteur à vecteur) |
| `hybrid_list_indexes` | Liste tous les index disponibles |
| `hybrid_index_info` | Détail d'un index (fichiers, chunks, modèle) |
| `hybrid_delete_index` | Supprime un index (confirmation requise) |
| `hybrid_list_drive` | Liste le contenu d'un dossier Drive |

**Points forts :** Isolation par client, warm start rapide (~7s), recherche multi-index, similarité documentaire.

**Limites :** Ressources locales (RAM selon le modèle d'embedding), pas de partage cloud natif.

---

## Réécriture de query (Query Rewriter)

Avant chaque retrieval, la query utilisateur est réécrite par `gemini-2.0-flash` pour améliorer la pertinence sémantique.

```
Query brute : "compare celio et fnac"
Query réécrite : "offres produits, services clients et positionnement commercial"
```

Le rewriter supprime les verbes d'action et garde les concepts. Il accepte un paramètre `context` (3 derniers échanges) pour résoudre les références conversationnelles :

```
Contexte : "Q: chiffrage celio"
Query : "combien sera facturé la prestation ?"
→ "tarifs facturation prestation Celio"
```

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
| `dense` | Similarité cosinus entre embeddings | Questions sémantiques, paraphrases |
| `sparse` | BM25 sur le texte brut | Codes produits, noms propres, jargon exact |
| `hybrid` | Fusion RRF dense + sparse (défaut) | Production générale |

Paramètres de fusion dans `hybrid/config.py` : `DENSE_WEIGHT=0.7`, `SPARSE_WEIGHT=0.3`, `RRF_K=60`.

---

## Stratégies de chunking

| Stratégie | Description | Usage |
|---|---|---|
| `fixed-128` | Chunks de 128 tokens, overlap 25 | Questions très précises, faits isolés |
| `fixed-256` | Chunks de 256 tokens, overlap 51 | FAQs, fiches produits |
| `fixed-512` | Chunks de 512 tokens, overlap 102 | Défaut recommandé |
| `fixed-1024` | Chunks de 1024 tokens, overlap 204 | Contrats, rapports longs |
| `semantic` | Coupure aux ruptures sémantiques (seuil 0.85) | CR de réunion, textes narratifs |
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
| **Comparaison RAG** | Vertex AI vs Hybrid en parallèle sur la même question, avec historique de session |
| **Index Manager** | Création, alimentation et suppression des index Hybrid |
| **Benchmark** | Visualisation des résultats benchmark (podium, leaderboard, graphiques, heatmap) |

---

## Structure des fichiers de données

```
hybrid/data/
├── hybrid.duckdb              # Base de données DuckDB (ignorée par git)
└── benchmark_results.json     # Résultats du dernier benchmark
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
FlagEmbedding          # BGE-M3
pymupdf                # Extraction PDF
langchain-text-splitters
scikit-learn

# UI
streamlit>=1.35
```
