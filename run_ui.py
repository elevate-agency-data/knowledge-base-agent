"""
Convenience launcher — runs the Streamlit UI from the project root.

Usage:
    python run_ui.py
    # or directly:
    streamlit run ui/app.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
ENTRY = ROOT / "ui" / "app.py"

subprocess.run(
    [sys.executable, "-m", "streamlit", "run", str(ENTRY)],
    cwd=str(ROOT),
)
