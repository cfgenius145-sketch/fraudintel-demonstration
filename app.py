import streamlit as st
import pandas as pd
from pipeline import score_claims

st.set_page_config(page_title="FraudIntel Demo", layout="wide")
st.title("Healthcare Fraud & Denial Intelligence — Demo")

st.write(
    "Upload a claims CSV (no PHI). Columns: "
    "`claim_id, member_id, provider_id, service_date, cpt_code, dx_code, place_of_service, units, amount, modifier`."
)

col1, col2 = st.columns(2)
with col1:
    use_llm = st.toggle("Add LLM explanations for top flagged claims (costs API tokens)", value=False)
with col2:
    llm_provider = st.selectbox("LLM provider", ["openai","anthropic"])

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    try:
        with st.spinner("Scoring claims..."):
            import os, platform, sys
            st.caption(f"Debug: Python {platform.python_version()} | pid {os.getpid()}")

            df_in = pd.read_csv(file)
            tmp = "/tmp/_claims.csv"; df_in.to_csv(tmp, index=False)

            # run the pipeline
            result = score_claims(tmp, use_llm=use_llm, llm_provider=llm_provider, top_k=50)

        st.success("Done.")
        st.caption("Sorted by risk_score (0–100). Use the download button below to export.")
        st.dataframe(
            result[["claim_id","provider_id","member_id","service_date","cpt_code",
                    "units","amount","risk_score","reasons","llm_deny_reason","llm_confidence","llm_rationale"]]
            .fillna("").head(500),
            use_container_width=True
        )
        st.download_button(
            "Download results CSV",
            result.to_csv(index=False).encode("utf-8"),
            file_name="fraudintel_results.csv",
            mime="text/csv"
        )
    except Exception as e:
        import traceback
        st.error("Something broke. Copy this error and paste it to me:")
        st.code("".join(traceback.format_exception(e)))


    st.success("Done.")
    st.caption("Sorted by risk_score (0–100). Use the download button below to export.")
    st.dataframe(
        result[["claim_id","provider_id","member_id","service_date","cpt_code",
                "units","amount","risk_score","reasons","llm_deny_reason","llm_confidence","llm_rationale"]]
        .fillna("").head(500),
        use_container_width=True
    )
    st.download_button(
        "Download results CSV",
        result.to_csv(index=False).encode("utf-8"),
        file_name="fraudintel_results.csv",
        mime="text/csv"
    )

with st.expander("CSV schema & tips"):
    st.markdown("""
- Dates: ISO format recommended (YYYY-MM-DD)  
- `place_of_service`: **11**=office, **22**=outpatient, **21**=inpatient  
- For best results, include realistic ranges for **amount** and **units**.  
    """)

