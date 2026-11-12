"""Settings in one place. Paths, limits and the business rules the docs define (docs/03, docs/04)."""

import datetime
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "var"
NYUMBA_DB = Path(os.getenv("NYUMBA_DB", DATA_DIR / "nyumba.db"))  # read-only for every analysis tool
AGENT_DB = Path(os.getenv("AGENT_DB", DATA_DIR / "agent.db"))  # outbox, feedback, run log

# "Today" is fixed so the synthetic data, metrics and tests agree. Override with NYUMBA_TODAY=YYYY-MM-DD.
TODAY = datetime.date.fromisoformat(os.getenv("NYUMBA_TODAY", "2026-10-25"))
HISTORY_MONTHS = 24

GRACE_DAYS = 5  # an invoice is "late" if paid in full after due_date + grace
ARREARS_TARGET = 0.10  # the goal: arrears rate below 10%

SQL_ROW_LIMIT = 200
SQL_TIMEOUT_S = 5.0
MAX_AGENT_STEPS = 6
LLM_MODEL = os.getenv("NYUMBA_MODEL", "gpt-4.1")  # any OpenAI chat model that supports tool calling
LLM_MAX_TOKENS = 8000
PANDAS_TIMEOUT_S = 10.0
PANDAS_MAX_CODE_CHARS = 4000
CHART_MAX_ROWS = 500
HIDDEN_COLUMNS = {"tenant": ["phone"]}  # contact details never reach the model (docs/02 privacy)
MPESA_PAYBILL = "400200"  # placeholder Paybill printed in reminder drafts (synthetic)
