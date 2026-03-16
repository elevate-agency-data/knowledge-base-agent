"""
Ajoute la racine du projet à sys.path, fixe le CWD, et charge les variables
d'environnement depuis rag_agent/.env.

Importé en premier dans chaque fichier Streamlit (app.py et pages/).
Garantit que rag_agent/, hybrid/ et ui/ sont tous importables,
que les chemins relatifs (ex: hybrid/data/hybrid.duckdb) fonctionnent,
et que GOOGLE_GENAI_USE_VERTEXAI / GOOGLE_APPLICATION_CREDENTIALS sont définis
avant que l'ADK ou le SDK genai ne soient importés.
"""

import os
import sys

# Racine = dossier parent de ui/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. Ajouter la racine à sys.path (une seule fois)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 2. Forcer le CWD à la racine pour que les chemins relatifs fonctionnent
#    (ex: hybrid/data/hybrid.duckdb, rag_agent/key.json)
if os.getcwd() != _ROOT:
    os.chdir(_ROOT)

# 3. Charger rag_agent/.env — contient GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_APPLICATION_CREDENTIALS…
#    Doit être fait AVANT tout import de google-adk / google-genai.
_ENV_FILE = os.path.join(_ROOT, "rag_agent", ".env")
if os.path.isfile(_ENV_FILE):
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_FILE, override=False)  # override=False : ne remplace pas les vars déjà définies
    except ImportError:
        # python-dotenv pas installé : chargement manuel
        with open(_ENV_FILE) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())
