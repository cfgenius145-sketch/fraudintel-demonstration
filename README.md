# FraudIntel — Healthcare Fraud & Denial Intelligence (Demo)

Prototype: upload a claims CSV → anomaly scores + heuristic reasons → optional LLM denial/fraud explanations.

## Quick start (local)

```bash
pip install -r requirements.txt
cp .env.example .env   # add your keys
python make_sample.py  # optional: creates sample_claims.csv
streamlit run app.py
