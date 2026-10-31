# NyumbaAgent

AI property analyst and manager agent for Kenyan landlords (AI unit assignment). Plan and design are in [docs/](docs/00_README.md).

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m nyumba_agent.data.generate --seed 42   # builds var/nyumba.db (synthetic, 24 months)
python -m pytest -q
```

All data is synthetic. Analysis tools only ever open `var/nyumba.db` read-only.

## Run the Ask tab
```bash
streamlit run nyumba_agent/ui/app.py                              # demo mode: scripted model, no key
NYUMBA_LLM=openai OPENAI_API_KEY=sk-... streamlit run nyumba_agent/ui/app.py   # live (or put the key in .env)
```
