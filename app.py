import streamlit as st
import pandas as pd
import tempfile
from pipeline import score_claims

st.set_page_config(page_title="FraudIntel Demo", layout="wide")
st.title("Healthcare Fraud & Denial Intelligence — Demo")

st.write(
    "Upload a claims CSV (no PHI). Required columns: "
    "`claim_id, member_id, provider_id, service_date, cpt_code, dx_code, "
    "place_of_service, units, amount, modifier`."
)

# --- Optional: LLM explanations (off by default to avoid API costs) ---
col1, col2 = st.columns(2)
with col1:
    use_llm = st.toggle(
        "Add LLM explanations for top flagged claims (optional; uses API credits)",
        value=False
    )
with col2:
    llm_provider = st.selectbox("LLM provider", ["openai", "anthropic"])

# --- File upload OR demo data ---
file = st.file_uploader("Upload CSV", type=["csv"])
use_demo = st.button("Or click here to try with demo data (synthetic)")

def make_demo_df(n=3000):
    import numpy as np, random, datetime as dt
    rng = np.random.default_rng(7)
    CPT = ["99213","99214","93000","80050","97110","97012","36415","J1885"]
    DX  = ["E11.9","I10","M54.5","J06.9","K21.9"]
    prov = [f"P{num:03d}" for num in range(1,21)]
    mem  = [f"M{num:05d}" for num in range(1,801)]
    rows=[]
    start=dt.date(2025,1,1)

    for i in range(n):
        p = random.choice(prov)
        m = random.choice(mem)
        d = start + dt.timedelta(days=int(rng.integers(0,120)))
        cpt = random.choice(CPT)
        units = max(1, int(abs(rng.normal(1.5,0.8))))
        base = {
            "99213":95,"99214":140,"93000":55,"80050":110,
            "97110":45,"97012":38,"36415":10,"J1885":25
        }[cpt]
        amt = round(max(10, float(rng.normal(base*units, base*0.25))), 2)
        mod = random.choice(["","25","59","76","80"])
        pos = random.choice(["11","22","21"])  # office, outpatient, inpatient
        dx  = random.choice(DX)
        rows.append([
            f"C{i:06d}", m, p, d.isoformat(), cpt, dx, pos, units, amt, mod
        ])

    # Inject some anomalies
    for _ in range(30):
        i = int(rng.integers(0, len(rows)))
        rows[i][7] = int(rows[i][7] * int(rng.integers(6, 10)))  # units spike
    for _ in range(25):
        i = int(rng.integers(0, len(rows)))
        rows[i][8] = round(rows[i][8] * float(rng.uniform(4.0, 8.0)), 2)  # amount spike

    return pd.DataFrame(rows, columns=[
        "claim_id","member_id","provider_id","service_date","cpt_code",
        "dx_code","place_of_service","units","amount","modifier"
    ])

df_in = None
if file is not None:
    df_in = pd.read_csv(file)

if df_in is None and use_demo:
    df_in = make_demo_df()

if df_in is not None:
    with st.spinner("Scoring claims..."):
        # Save to a temp CSV so the pipeline can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            df_in.to_csv(tmp.name, index=False)
            result = score_claims(
                tmp.name,
                use_llm=use_llm,
                llm_provider=llm_provider,
                top_k=50
            )

    st.success("Done. Showing top results by risk_score.")
    cols_to_show = [
        "claim_id","provider_id","member_id","service_date","cpt_code",
        "units","amount","risk_score","reasons","llm_deny_reason",
        "llm_confidence","llm_rationale"
    ]
    show = [c for c in cols_to_show if c in result.columns]
    st.dataframe(result[show].head(500), use_container_width=True)

    csv_bytes = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results CSV",
        data=csv_bytes,
        file_name="fraudintel_results.csv",
        mime="text/csv"
    )

with st.expander("CSV schema & tips"):
    st.markdown(
        "- Dates: ISO format recommended (YYYY-MM-DD)\n"
        "- `place_of_service`: **11**=office, **22**=outpatient, **21**=inpatient\n"
        "- Include realistic ranges for **amount** and **units** for better signals.\n"
        "- LLM explanations are optional; leave off if you don’t have API keys."
    )
