# Knowledge Base Agent

Base de connaissance documentaire interrogeable en langage naturel, avec
recherche hybride (sémantique + mots exacts) sur DuckDB local, un agent
conversationnel Google ADK et une interface Streamlit.

L'application est **multi-marque** : un seul fichier de configuration
(`shared/brand.py`) porte l'identité, le thème, les prompts, les rôles et la
base de données de chaque profil. Changer `BRAND_PROFILE` change toute
l'application, sans toucher au code.

---

## Architecture

```
knowledge-base-agent/
├── shared/
│   ├── brand.py            # ★ Registre des profils de marque — source unique
│   │                       #   identité, thème, polices, prompts, rôles, BDD,
│   │                       #   taxonomie SAV image, corpus de démonstration
│   ├── role_permissions.py # RBAC : par index (nom) et par chunk (audience_role)
│   ├── index_resolver.py   # Choix automatique des index via Gemini Flash
│   ├── query_rewriter.py   # Réécriture sémantique des questions
│   └── gen_streamlit_config.py  # Génère .streamlit/config.toml depuis la marque
│
├── hybrid/                 # Pipeline de recherche (local, DuckDB)
│   ├── config.py           # Chunking, embeddings, RRF, chemins (issus de brand)
│   ├── embeddings/         # minilm-384, mpnet-768, e5-base/large, bge-m3, vertex
│   ├── ingestion/
│   │   ├── atelier_map.py  # Chemin de dossier → index + dimensions (partagé)
│   │   ├── local_ingest.py # Ingestion depuis le disque
│   │   ├── extractor.py    # Extraction Drive (PDF, Docx, Xlsx, Pptx, Google Docs)
│   │   ├── chunker.py      # Découpe par tokens du modèle d'embedding
│   │   └── metadata.py     # Langue, domaine, étiquettes, dimensions atelier
│   ├── retrieval/          # Dense (HNSW), Sparse (BM25), fusion RRF, filtres
│   ├── stores/             # DuckDB (FLOAT[768] + VSS/FTS) et AlloyDB
│   └── tools/              # Outils ADK : hybrid_query, hybrid_add_data_atelier…
│
├── vision/                 # Analyse d'image SAV (Gemini via Vertex AI)
│   ├── sav.py              # Lecture d'une photo : produit, dommages, éléments
│   ├── uploads.py          # Stockage des photos + références cloisonnées par user
│   ├── client.py           # Client google-genai + reprise sur quota
│   └── config.py           # Modèle, tailles d'image, dossier de dépôt
│
├── rag_agent/              # Agent conversationnel (Google ADK)
│   ├── agent.py            # Instruction + outils (lecture / admin / image)
│   ├── runtime_context.py  # Rôle et identité de l'appelant, hors de portée du prompt
│   └── tools/              # sav_image_tools, get_document_content, compare_documents…
│
├── ui/                     # Interface Streamlit
│   ├── app.py              # Router, warmup, purge des dépôts
│   ├── auth.py             # Authentification PBKDF2, une base de comptes par marque
│   ├── pages/              # Login, Accueil, How RAG Works, RAG Demo,
│   │                       # Knowledge Base, Simple Chat, Agent Chat, Administration
│   ├── components/
│   │   ├── lux_style.py    # ★ Langage visuel partagé (piloté par la marque)
│   │   ├── sav_sheet.py    # Fiche d'atelier — rendu d'une analyse photo
│   │   ├── detail_panel.py # Panneau unique : sources, passages, outils
│   │   └── …               # chat_message, answer_renderer, sidebar_auth, brand_header
│   └── services/           # agent_runner, hybrid_service, chat_store
│
├── scripts/
│   ├── ingest_local.py     # Ingérer un corpus depuis le disque
│   ├── ingest_drive.py     # Ingérer l'arborescence Drive de la marque
│   ├── diag_bm25.py        # Diagnostiquer la recherche plein texte
│   └── seed_admin.py       # Créer le premier compte administrateur
│
├── docs/                   # embedding_models.md, business_metadata.md
├── run_ui.py
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
MODEL                   = "gemini-2.5-flash"   # ou gemini-2.5-pro
DEFAULT_TOP_K           = 10
DEFAULT_DISTANCE_THRESHOLD = 0.5
DEFAULT_EMBEDDING_MODEL = "publishers/google/models/text-embedding-005"
```

### 4. Paramètres Hybrid RAG (`hybrid/config.py`)

```python
ENV                     = "local"       # "local" → DuckDB | "gcp" → AlloyDB
DEFAULT_EMBEDDING_MODEL = "e5-base-768"
CHUNK_SIZE              = 320           # en tokens — voir « Chunking » plus bas
CHUNK_OVERLAP           = 32            # en tokens
DENSE_WEIGHT            = 0.7
SPARSE_WEIGHT           = 0.3
RRF_DENSE_GATED         = False         # True = écarte les trouvailles BM25 seules
DEDUP_CONTENT           = True          # fusionne les passages identiques

# DUCKDB_PATH et DRIVE_ROOT_FOLDER ne sont pas définis ici : ils viennent
# du profil de marque actif (shared/brand.py), pour que chaque client garde
# sa propre base et son propre dossier Drive.
```

`DRIVE_ROOT_FOLDER` est le point d'entrée de toutes les recherches Drive. La recherche de dossiers clients est toujours restreinte à ses enfants directs, évitant les collisions de noms entre clients.

---

## Profils de marque

Tout ce qui distingue un client vit dans `shared/brand.py` : nom, sous-titre,
thème et polices, fragments de prompt, rôles et règles d'accès, dossier Drive,
fichier DuckDB, base de comptes. Quatre profils sont fournis — `hermes`,
`indica`, `lacoste`, `activate` — sélectionnés par `BRAND_PROFILE`
(défaut : `hermes`).

Chaque profil a **sa propre base de connaissance et ses propres comptes** : les
données ne se mélangent jamais d'un client à l'autre.

Ajouter une marque = copier un bloc `BrandProfile(...)`, l'enregistrer dans
`PROFILES`. Rien d'autre à modifier.

---

## Lancer l'interface Streamlit

```cmd
set BRAND_PROFILE=hermes
rag_venv\Scripts\python.exe -m shared.gen_streamlit_config
rag_venv\Scripts\streamlit.exe run ui\app.py
```

```bash
# bash / Linux
BRAND_PROFILE=hermes python -m shared.gen_streamlit_config
BRAND_PROFILE=hermes streamlit run ui/app.py
```

La deuxième ligne génère `.streamlit/config.toml` (couleurs et polices de la
marque). **À rejouer à chaque changement de `BRAND_PROFILE`**, sinon Streamlit
conserve le thème précédent.

L'interface s'ouvre sur `http://localhost:8501`.

> **Une seule instance à la fois.** DuckDB n'autorise qu'un processus écrivain :
> une seconde application lancée en parallèle verra une base vide, sans erreur
> explicite.

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

## Ingestion « atelier » — un index par domaine, le reste en métadonnées

Un document se qualifie sur plusieurs axes : domaine, ligne de produit,
matière, zone géographique, public destinataire. Les encoder tous dans le nom
de l'index ferait exploser leur nombre.

**Seul l'axe sur lequel on route porte le nom de l'index** — le domaine, parce
que c'est de lui que parle la question. Les autres dimensions deviennent des
colonnes filtrables. Ajouter une zone ou une ligne produit devient une valeur,
pas une restructuration.

L'arborescence Drive reste hiérarchique pour ceux qui y déposent ; l'ingestion
l'aplatit :

```
Rag_hermes / <LigneProduit> / <Domaine> / [<Zone>] / fichier
              │               │            └ zone            → métadonnée
              │               └ domaine                      → NOM D'INDEX
              └ ligne_produit  (« _Transverse » = aucune)     → métadonnée
_meta.yaml (audience_role, matiere) — hérité, le plus proche l'emporte
```

Colonnes ajoutées au schéma : `ligne_produit`, `zone`, `audience_role[]`,
`matiere[]`. Toutes nullables, avec migration automatique des tables
existantes : les autres profils ne changent pas de comportement.

```cmd
rag_venv\Scripts\python.exe -m scripts.ingest_drive            :: depuis Drive
rag_venv\Scripts\python.exe -m scripts.ingest_local "<chemin>" :: depuis le disque
```

Les deux sources partagent le même mapping (`hybrid/ingestion/atelier_map.py`),
donc produisent des index identiques.

---

## Contrôle d'accès (RBAC)

Deux niveaux, tous deux définis par marque dans `shared/brand.py` :

| Niveau | Champ | Portée |
|---|---|---|
| Par index | `role_keywords` | mots-clés testés sur le nom de l'index |
| Par chunk | `role_audience` | valeurs `audience_role` portées par le passage |

Le second est le plus fin : un même index peut contenir des passages réservés à
l'atelier et d'autres ouverts aux conseillers. Un rôle inconnu retombe sur le
rôle par défaut — jamais sur un accès total.

Le rôle de l'appelant transite par `rag_agent/runtime_context.py`, posé par le
runner sur le fil d'exécution : le prompt de l'agent ne peut pas le contourner.

---

## Analyse d'image SAV

Une photo déposée dans le chat est lue avant la recherche : quel produit, quels
dommages, quels éléments récupérables. Le vocabulaire vient du profil de marque
(`vision_product_lines`, `vision_materials`, `vision_damage_types`,
`vision_components`), pas du modèle.

Le résultat est présenté comme une **fiche d'atelier**, puis l'agent enchaîne
sur la base documentaire pour la procédure, la garantie et les délais.

Frontière volontaire : la photo établit des **observations** ; tarif, délai,
couverture de garantie et verdict d'authenticité restent tirés des documents.

Les outils (`sav_analyze_image`, `sav_list_images`) ne sont enregistrés que pour
les marques ayant défini cette taxonomie — sinon l'agent n'annonce pas une
capacité qu'il refuserait.

---

## Résolution automatique des index

Avant chaque requête multi-index, `shared/index_resolver.py` détermine via Gemini Flash quels index interroger en fonction de la query :

- Si la query mentionne un client précis → index correspondant uniquement
- Si la query est générale ou comparative → tous les index

Utilisé par l'outil `hybrid_query`, la page Simple Chat et l'agent.

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

Le chunking est réalisé en **tokens réels**, via le tokenizer HuggingFace du
modèle d'embedding — aucune approximation en caractères.

| Stratégie | Description | Usage |
|---|---|---|
| `fixed` (défaut) | 320 tokens, overlap 32 | Production générale |
| `semantic` | Coupure aux ruptures de sens (seuil cosinus 0.85) | Comptes rendus, textes narratifs |
| `hierarchical` | Chunks enfants 256 + parents 1024 | Documents longs multi-niveaux |

**Pourquoi 320 et non le maximum du modèle.** Avec `CHUNK_SIZE = 0`, le
découpage prend le `max_seq_length` du modèle (510 pour `e5-base-768`). C'est
le **plafond encodable**, pas l'optimum : un chunk plein comprime ~380 mots
dans un seul vecteur de 768 dimensions, et une question précise remonte alors
un bloc majoritairement hors sujet. Mesuré sur le corpus hermès : 510 tokens
donnent 186 chunks (~2,5 par document), 320 en donnent 285 (~3,9) — une
granularité nettement meilleure pour un surcoût négligeable.

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

Les résultats sont écrits dans `hybrid/data/benchmark_results.json`.

---

## Pages de l'interface Streamlit

| Page | Description |
|---|---|
| **Accueil** | Ce que fait l'application, ce que contient la base |
| **How RAG Works** | Explication animée : d'un document déposé à une réponse citée |
| **RAG Demo** | La même question par le sens, par les mots exacts, puis combinée — scores et passages à l'appui |
| **Knowledge Base** | Création, alimentation et suppression des index |
| **Simple Chat** | Échange direct, avec ou sans consultation de la base |
| **Agent Chat** | L'agent choisit les index, cite ses sources ; photo joignable au message |
| **Administration** | Gestion des comptes et des rôles (admin) |

L'apparence est unifiée par `ui/components/lux_style.py`, injecté depuis la
barre latérale : aucune page ne peut diverger en oubliant un appel. Couleurs et
polices proviennent du profil actif, donc chaque marque hérite de la même
structure dans sa propre palette.

---

## Structure des fichiers de données

Tout est ignoré par git, et **cloisonné par marque** : chaque profil déclare son
propre `duckdb_path` et son propre `users_db_path`.

```
hybrid/data/
├── hermes_hybrid.duckdb       # Base de connaissance du profil hermes
├── hybrid.duckdb              # Profil indica
└── benchmark_results.json

Activate_db/
└── lacoste_hybrid.duckdb      # Profils lacoste et activate

ui/data/
├── hermes_users.db            # Comptes du profil hermes (SQLite, PBKDF2)
├── users.db                   # Comptes des profils indica / activate
├── uploads/                   # Photos déposées dans le chat (purgées au démarrage)
├── chat/chat.duckdb           # Historique des conversations
└── agent_display/

.streamlit/config.toml         # Thème généré depuis shared/brand.py
```

Le premier compte administrateur est créé automatiquement à la création d'une
base de comptes, à partir de `DEFAULT_ADMIN_EMAIL` / `DEFAULT_ADMIN_NAME` /
`DEFAULT_ADMIN_PASSWORD` (dans `rag_agent/.env`, jamais en dur). Sans mot de
passe renseigné, aucun compte n'est créé.

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
