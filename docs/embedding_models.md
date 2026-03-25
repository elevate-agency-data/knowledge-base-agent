# Embedding Models — Documentation & Choix Techniques

## Contexte

Le pipeline hybrid RAG utilise des modèles d'embedding pour convertir le texte en vecteurs numériques.
Ces vecteurs permettent la **recherche sémantique dense** : deux textes similaires dans le sens auront des vecteurs proches, même s'ils ne partagent pas les mêmes mots-clés.

Le choix du modèle impacte directement :
- La **qualité de la recherche** (pertinence des chunks retournés)
- La **vitesse d'ingestion** (temps pour vectoriser les documents)
- La **consommation mémoire** (RAM nécessaire au runtime)
- Le **support multilingue** (français, anglais, autres langues)

---

## Les 5 modèles disponibles

### 1. `minilm-384` — MiniLM-L12-v2 Multilingual
```
Modèle    : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Dimension : 384
Taille    : ~120 MB
Langues   : 50+
```

**Pourquoi ce modèle ?**
- Distillation de BERT large en un modèle 6x plus petit sans perte majeure de qualité
- Architecture L12 (12 couches) offre un bon compromis profondeur/vitesse
- Entraîné sur des paires de paraphrases multilingues → excellente similarité sémantique

**Points forts**
- Le plus rapide des 5 modèles (~3-5x plus rapide que mpnet-768)
- Faible consommation mémoire — tourne même sur machine sans GPU
- Suffisant pour des documents courts et requêtes simples

**Limites**
- Dimension 384 = vecteurs moins expressifs que 768 ou 1024
- Perd en précision sur des documents longs ou des concepts complexes
- Moins performant sur les nuances sémantiques fines

**Usage recommandé** : Prototypage rapide, tests, environnements contraints en ressources

---

### 2. `mpnet-768` — MPNet Multilingual "Défaut actuel"
```
Modèle    : sentence-transformers/paraphrase-multilingual-mpnet-base-v2
Dimension : 768
Taille    : ~420 MB
Langues   : 50+
```

**Pourquoi ce modèle ?**
- MPNet combine les avantages de BERT (masked language modeling) et XLNet (autoregressive)
  → meilleure modélisation des dépendances entre tokens que BERT seul
- La variante `paraphrase-multilingual` est fine-tunée spécifiquement pour la similarité sémantique
  sur des corpus multilingues (données MS MARCO, AllNLI, etc.)
- Dimension 768 = standard optimal entre expressivité et performance

**Points forts**
- Excellent rapport qualité/vitesse pour des documents professionnels
- Très bon français et anglais (cas d'usage principal : docs clients Insight Factory) 
 

**Limites**
- Ne génère pas de vecteurs sparse → le sparse search utilise BM25 textuel
- Légèrement moins précis que e5-large sur les textes très techniques

**Usage recommandé** : Production stable, cas d'usage général, docs clients FR/EN

---

### 3. `e5-large-1024` — E5 Large Multilingual
```
Modèle    : intfloat/multilingual-e5-large
Dimension : 1024
Taille    : ~1.3 GB
Langues   : 100+
```

**Pourquoi ce modèle ?**
- E5 (EmbEddings from bidirEctional Encoder rEpresentations) est entraîné avec une méthode
  contrastive sur des paires query/passage annotées — spécifiquement optimisé pour le RAG
- Le préfixe `"query: "` / `"passage: "` permet au modèle de distinguer requête et document,
  ce qui améliore significativement la pertinence en retrieval
- Dimension 1024 = vecteurs très expressifs, capturent des nuances sémantiques fines

**Points forts**
- Meilleure qualité de retrieval parmi les modèles locaux sans GPU
- Couverture de 100+ langues dont les langues moins représentées
- Architecture "query-aware" — conçu nativement pour le RAG

**Limites**
- 1.3 GB en RAM → nécessite une machine correctement dimensionnée
- Plus lent que mpnet-768 (~2x)
- Les préfixes query/passage sont obligatoires (gérés automatiquement dans le code)

**Usage recommandé** : Quand la qualité prime sur la vitesse, documents techniques complexes

---

### 4. `bge-m3` — BGE-M3 (BAAI)  "Recommandé long terme"
```
Modèle    : BAAI/bge-m3
Dimension : 1024 (dense) + sparse (lexical weights)
Taille    : ~2.3 GB
Langues   : 100+
Librairie : FlagEmbedding
```

**Pourquoi ce modèle ?**
- BGE-M3 signifie **Multi-lingual, Multi-functionality, Multi-granularity**
- Unique parmi les modèles open-source : génère **dense ET sparse en un seul forward pass**
  → élimine le besoin d'un modèle BM25 séparé pour le sparse
- Les vecteurs sparse BGE-M3 sont des **poids lexicaux appris** (≠ BM25 statistique)
  → meilleure précision que BM25 car les poids sont contextuels
- Entraîné sur 70+ langues avec des corpus massifs (C-PACK dataset)
- SOTA (State of the Art) sur les benchmarks BEIR et MTEB multilingues

**Architecture dense+sparse**
```
Texte → BGE-M3 → {
    dense  : float[1024]          # similarité sémantique
    sparse : {token_id: weight}   # poids lexicaux appris
}
```

La fusion RRF bénéficie alors de deux signaux complémentaires de haute qualité :
- Dense : capture le sens, les synonymes, les paraphrases
- Sparse BGE-M3 : capture les termes exacts avec pondération contextuelle

**Points forts**
- Meilleure qualité globale de retrieval du pipeline
- Sparse "intelligent" vs BM25 statistique des autres modèles
- Un seul modèle pour les deux types de recherche (économie de ressources)

**Limites**
- 2.3 GB en RAM — requiert une machine bien dimensionnée ou GPU
- Ingestion plus lente que mpnet-768 traitement du texte necessaire

**Usage recommandé** 

---

### 5. `vertex` — Google text-embedding-005
```
Modèle    : publishers/google/models/text-embedding-005
Dimension : 768
Hébergement : Google Cloud Vertex AI (cloud)
Langues   : 100+
```

**Pourquoi ce modèle ?**
- Modèle propriétaire Google, entraîné sur des données massives et optimisé pour Vertex AI RAG
- Intégration native avec l'écosystème GCP (AlloyDB pgvector, Vertex AI Search)
- Pas de RAM locale requise — l'inférence tourne dans le cloud

**Points forts**
- Aucune consommation de ressources locales
- Qualité comparable à e5-large sur les tâches en anglais/français
- Idéal pour la mise en production sur GCP avec AlloyDB

**Limites**
- Coût par requête (facturation Vertex AI)
- Rate limiting géré par `EMBEDDING_REQUESTS_PER_MIN` dans config.py
- Non disponible en mode local/offline

**Usage recommandé** : Production GCP avec AlloyDB, quand ENV = "gcp"

---

## Comparatif synthétique

| Modèle | Dim | RAM | Vitesse | Qualité FR/EN | Sparse natif | Offline |
|---|---|---|---|---|---|---|
| `minilm-384` | 384 | ~200MB | 4/4 | 2/4 | Non | Oui |
| `mpnet-768` | 768 | ~600MB | 3/4 | 3/4 | Non | Oui |
| `e5-large-1024` | 1024 | ~1.5GB | 2/4 | 3/4 | Non | Oui |
| `bge-m3` | 1024 | ~2.5GB | 2/4 | 4/4 | Oui | Oui |
| `vertex` | 768 | 0 | 3/4 | 4/4 | Non | Non |

---

## Recommandations par scénario

| Scénario | Modèle recommandé |
|---|---|
| Développement / tests rapides | `minilm-384` |
| Production stable (actuel) | `mpnet-768` |
| Meilleure qualité sans GPU | `e5-large-1024` |
| Meilleure qualité globale (objectif) | `bge-m3` |
| Production GCP (ENV=gcp) | `vertex` |
| Machine avec GPU | `bge-m3` |

---

## Paramètres de configuration (`hybrid/config.py`)

```python
DEFAULT_EMBEDDING_MODEL: str = "mpnet-768"    
VERTEX_EMBEDDING_MODEL:  str = "publishers/google/models/text-embedding-005"
EMBEDDING_REQUESTS_PER_MIN: int = 1000    
```

> **Important** : Tous les chunks d'un même index doivent être vectorisés avec le même modèle.
> Si changement de modèle, l'index est a supprimé et re-crée de zéro.

---

## Fonctionnement du hybrid retrieval

Quel que soit le modèle dense choisi, la recherche hybrid combine toujours :

```
Query
  ├── Dense  : embedding_model.embed_query(query) → cosine similarity
  └── Sparse : BM25 natif DuckDB sur le champ content TEXT
        (ou sparse BGE-M3 si bge-m3 est activé)

Fusion RRF :
  score = DENSE_WEIGHT × 1/(k + rank_dense)
        + SPARSE_WEIGHT × 1/(k + rank_sparse)

Paramètres (hybrid/config.py) :
  DENSE_WEIGHT  = 0.7   # 70% sémantique
  SPARSE_WEIGHT = 0.3   # 30% BM25
  RRF_K         = 60    # constante de lissage
```

La pondération 70/30 dense/sparse est le standard de la littérature RAG.
Elle peut être ajustée selon le type de documents :
- Documents très techniques avec jargon → augmenter SPARSE_WEIGHT (ex: 0.4)
- Documents narratifs / paraphrasés → augmenter DENSE_WEIGHT (ex: 0.8)

---

## Modes de retrieval

### Dense
Recherche par **similarité sémantique vectorielle** (cosinus entre vecteurs).

Le modèle d'embedding convertit la query et chaque chunk en vecteur. Les chunks dont le vecteur est le plus proche de celui de la query sont retournés — même si aucun mot ne correspond exactement.

**Quand c'est pertinent :**
- Questions formulées différemment du document source
- Synonymes, paraphrases, langues mélangées
- Questions conceptuelles

**Exemple :**
> Query : *"comment fonctionne le parcours d'accueil ?"*
> Chunk retrouvé : *"Le processus d'onboarding débute le premier jour..."*
> → Pas un seul mot en commun, mais retrouvé grâce au sens

**Limite :** Rate sur les noms propres, codes produits, acronymes spécifiques au client.

---

### Sparse (BM25)
Recherche par **correspondance de mots-clés** (fréquence des termes dans le document).

BM25 (Best Match 25) est une évolution de TF-IDF : il pondère les termes par leur fréquence dans le chunk et leur rareté dans le corpus. Pas d'embedding — purement textuel.

**Quand c'est pertinent :**
- Recherche de références exactes : codes SKU, noms de produits, identifiants
- Termes très spécifiques qui n'ont pas d'équivalent sémantique
- Documents très techniques avec jargon métier

**Exemple :**
> Query : *"procédure onboarding RH"*
> → Le dense pourrait rater si le terme exact n'a pas de sens vectoriel
> → Le sparse retrouve les chunks qui contiennent exactement ce terme

**Limite :** Rate les paraphrases et les questions formulées autrement que dans le document.

---

### Hybrid (RRF)
**Combinaison de dense + sparse** via Reciprocal Rank Fusion.

Les deux listes de résultats sont fusionnées en pondérant les rangs : un chunk bien classé par les deux méthodes remonte encore plus haut. C'est le mode recommandé par défaut.

**Quand c'est pertinent :**
- Corpus mixte (documents narratifs + tableaux techniques)
- Quand on ne sait pas à l'avance si la query sera sémantique ou lexicale
- Production générale

**Exemple :**
> Query : *"politique de retour produit défectueux réf. 4872"*
> → Dense retrouve les chunks sur la politique de retour (sens)
> → Sparse retrouve les chunks mentionnant "4872" (mot-clé exact)
> → RRF combine les deux → résultat optimal

**Paramètres dans `hybrid/config.py` :**
```python
DENSE_WEIGHT  = 0.7   # 70% sémantique
SPARSE_WEIGHT = 0.3   # 30% BM25
RRF_K         = 60    # constante de lissage (standard littérature)
```

---

## Stratégies de chunking

### Fixed (fixe)
Découpe le texte en chunks de **taille fixe avec chevauchement**.

```
Texte : [--------512 tokens--------][overlap 102][--------512 tokens--------]
```

Le chevauchement évite de couper une information importante à la frontière de deux chunks.
L'overlap est calculé automatiquement à **20% de la taille** du chunk.

**Tailles testées en benchmark :**

| Taille | Overlap (20%) | Cas d'usage typique |
|---|---|---|
| **128** | 25 | Questions très précises, chunks courts, réponses factuelles |
| **256** | 51 | Paragraphes courts, FAQs, fiches produits |
| **512** | 102 | Défaut recommandé — bon équilibre contexte/précision |
| **1024** | 204 | Documents longs nécessitant beaucoup de contexte (contrats, rapports) |

**Règle générale :**
- Chunk **petit** (128-256) → meilleure précision, moins de contexte → bon pour les faits isolés
- Chunk **grand** (512-1024) → plus de contexte, moins de précision → bon pour les questions de compréhension

**Quand c'est pertinent :**
- Corpus homogène avec paragraphes de taille similaire
- Documents structurés (rapports, fiches produits, SOPs)
- Quand la vitesse d'ingestion compte

**Exemple :**
> Document : guide de 10 pages sur la politique RH
> → fixed-256 : ~N chunks de 256 tokens → chaque chunk = 1 paragraphe
> → fixed-512 : ~N/2 chunks de 512 tokens → chaque chunk = 1 à 2 paragraphes
> → fixed-1024 : ~N/4 chunks de 1024 tokens → chaque chunk = 1 section entière

**Résultat benchmark :** Meilleur mode sur le dataset synthétique. Lancer le benchmark pour identifier la taille optimale sur ton corpus.

---

### Semantic (sémantique)
Découpe le texte aux **ruptures sémantiques** détectées par le modèle d'embedding.

Au lieu de couper à taille fixe, le chunker calcule la similarité entre phrases consécutives. Quand la similarité chute en dessous du seuil (`SEMANTIC_BREAKPOINT_THRESHOLD = 0.85`), il coupe — le sujet a changé.

**Quand c'est pertinent :**
- Documents avec sections de longueur très variable (CR de réunion, emails)
- Textes narratifs sans structure claire (articles, comptes-rendus)
- Quand on veut que chaque chunk corresponde à une idée complète

**Exemple :**
> Document : compte-rendu de réunion avec sujets variés
> → Chunk 1 : tout ce qui concerne le budget (3 paragraphes)
> → Chunk 2 : tout ce qui concerne le planning (1 paragraphe)
> → Chunk 3 : tout ce qui concerne les décisions RH (5 paragraphes)
> → Frontières naturelles selon le contenu, pas selon la taille

**Limite :** Plus lent à l'ingestion (calcule des embeddings pendant le chunking), taille des chunks imprévisible.

---

### Hierarchical (hiérarchique)
Crée **deux niveaux de chunks** : des chunks parents larges et des chunks enfants fins.

```
Parent (1024 tokens) : contexte global
  └── Enfant 1 (256 tokens) : détail 1
  └── Enfant 2 (256 tokens) : détail 2
  └── Enfant 3 (256 tokens) : détail 3
```

La recherche se fait sur les chunks enfants (plus précis), mais le contexte du parent est disponible pour la génération.

**Quand c'est pertinent :**
- Documents très longs où le contexte global compte (contrats, rapports annuels)
- Questions qui nécessitent un contexte large pour être bien répondues
- Quand la précision du retrieval ET la richesse du contexte sont importantes

**Exemple :**
> Document : contrat de 50 pages
> → Parent : clause complète sur la résiliation (1024 tokens)
> → Enfant : sous-section "conditions de résiliation anticipée" (256 tokens)
> Query : *"conditions de résiliation anticipée"*
> → Retrieval trouve l'enfant précis, génération utilise le contexte du parent

**Limite :** Plus de chunks stockés (N×4 vs fixed), recall@10 élevé mais MRR plus faible (le bon chunk est trouvé mais pas toujours en premier).

---

## Métriques de benchmark

Le benchmark évalue chaque combinaison sur 4 métriques complémentaires.

### MRR — Mean Reciprocal Rank
**"Le bon chunk est-il en premier ?"**

Mesure à quelle position apparaît le premier chunk pertinent dans les résultats.

```
MRR = 1 / rang_du_premier_chunk_pertinent

Chunk pertinent en position 1 → MRR = 1.0    (parfait)
Chunk pertinent en position 2 → MRR = 0.5
Chunk pertinent en position 5 → MRR = 0.2
Pas trouvé dans le top-K      → MRR = 0.0
```

**Quand c'est important :** Quand Gemini n'utilise que les premiers résultats pour générer la réponse. Un MRR élevé = la réponse est dans les premiers chunks retournés.

---

### nDCG — Normalized Discounted Cumulative Gain
**"Les bons chunks sont-ils bien classés ?"**

Mesure la qualité du **classement global** : un chunk pertinent en position 1 rapporte plus qu'en position 5. Normalisé entre 0 et 1 par rapport au classement idéal.

```
DCG  = Σ relevance(i) / log2(i + 1)    # pénalise les bons chunks en bas du classement
nDCG = DCG / DCG_idéal                 # normalisé → toujours entre 0 et 1
```

**Quand c'est important :** Métrique principale du benchmark — reflète fidèlement la qualité globale du retrieval. Un nDCG de 1.0 = classement parfait.

---

### Recall@K
**"Tous les bons chunks sont-ils retrouvés dans le top-K ?"**

Mesure le **taux de rappel** : sur tous les chunks pertinents existants, combien sont dans les K premiers résultats ?

```
Recall@10 = chunks_pertinents_dans_top10 / total_chunks_pertinents

2 chunks pertinents existent, 2 retrouvés dans le top 10 → Recall = 1.0
2 chunks pertinents existent, 1 retrouvé dans le top 10  → Recall = 0.5
```

**Quand c'est important :** Pour des questions qui nécessitent plusieurs sources (ex: "compare X et Y"). Un Recall élevé avec un MRR faible = tous les bons chunks sont là mais mal classés.

---

### Precision@K
**"Parmi les K résultats retournés, combien sont pertinents ?"**

```
Precision@10 = chunks_pertinents_dans_top10 / 10

3 chunks pertinents dans les 10 premiers → Precision = 0.3
```

**Quand c'est important :** Limite le bruit envoyé à Gemini — une précision faible = beaucoup de chunks hors-sujet dans le contexte.

---

### Résumé visuel

```
Résultats retournés : [✅, ❌, ✅, ❌, ❌, ✅, ❌, ❌, ❌, ❌]
                       pos1  pos2  pos3             pos6

MRR        = 1/1    = 1.00   (premier résultat pertinent en pos 1)
Recall@10  = 3/3    = 1.00   (les 3 chunks pertinents sont tous retrouvés)
Precision@10 = 3/10 = 0.30   (30% des résultats retournés sont pertinents)
nDCG@10    ≈ 0.75           (pénalisé car pos3 et pos6 ne sont pas en haut)
```

---

## Lancer le benchmark

```bash
# Dataset synthétique intégré (aucun prérequis)
python -m hybrid.benchmark.eval_retrieval

# Avec un fichier Q&A sur tes vrais documents
python -m hybrid.benchmark.eval_retrieval --mode qa --qa-file mon_dataset.csv
```

**Format du fichier Q&A (CSV ou Excel) :**

| Q | A | url |
|---|---|---|
| Quelle est la politique de retour ? | Les retours sont acceptés sous 30 jours... | https://docs.google.com/... |

Le benchmark teste **90 combinaisons** :

```
5 embeddings × (4 tailles fixed + 1 semantic + 1 hierarchical) × 3 retrieval
= 5 × 6 × 3 = 90 combinaisons
```

| Axe | Valeurs |
|---|---|
| Embeddings | minilm-384, mpnet-768, e5-large-1024, bge-m3, vertex |
| Chunking fixed | 128, 256, 512, 1024 tokens (overlap = 20%) |
| Chunking autre | semantic, hierarchical |
| Retrieval | dense, sparse, hybrid |

Les résultats sont sauvegardés dans `hybrid/data/benchmark_results.json` et visualisables dans la page **Benchmark** de l'UI Streamlit.
