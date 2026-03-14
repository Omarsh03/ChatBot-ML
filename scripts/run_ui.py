import streamlit.web.cli as stcli
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


if __name__ == "__main__":
    app_path = PROJECT_ROOT / "ui" / "streamlit_app.py"
    sys.argv = ["streamlit", "run", str(app_path)]
    raise SystemExit(stcli.main())
