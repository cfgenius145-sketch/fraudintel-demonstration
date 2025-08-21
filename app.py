import streamlit as st
import pandas as pd
import numpy as np
from dateutil.parser import isoparse
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import traceback

st.set_page_config(page_title="FraudIntel — Minimal Demo", layout="wide")
st.title("FraudIntel — Minimal Claims Risk Demo")

st.write("Upload a **CSV** with these columns (no PHI): "
         "`claim_id, member_id, provider_id, service_date, cpt_code, dx_code, place_of_service, units, amount, modifier`.")

# ---------- helpers ----------
def _safe_dt(x):
    try:
        return isoparse(str(x)).date()
    except Exception:
        return pd.NaT

def featurize(df: pd.DataFrame):
    df = df.copy()

    # basic types
    df["service_date"] = df["service_date"].apply(_safe_dt)
    df["dow"] = df["service_date"].apply(lambda d: d.weekday() if pd.notna(d) else -1)
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)

    # numeric
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    # provider stats
    gP = df.groupby("provider_id").agg(
        prov_amt_median=("amount","median"),
        prov_amt_p90=("amount", lambda s: np.percentile(s,90) if len(s)>0 else 0),
        prov_units_median=("units","median"),
        prov_claims=("claim_id","count")
    )
    df = df.join(gP, on="provider_id")

    # CPT stats
    gC = df.groupby("cpt_code").agg(
        cpt_amt_median=("amount","median"),
        cpt_units_median=("units","median"),
        cpt_count=("claim_id","count")
    )
    df = df.join(gC, on="cpt_code")

    # provider–CPT rarity
    gPC = df.groupby(["provider_id","cpt_code"]).agg(pcpt_count=("claim_id","count"))
    df = df.merge(gPC, on=["provider_id","cpt_code"], how="left")

    # same-day member density
    gMD = df.groupby(["member_id","service_date"]).size().rename("member_day_claims")
    df = df.join(gMD, on=["member_id","service_date"])
    df["member_day_claims"] = df["member_day_claims"].fillna(1)

    # features
    eps = 1e-6
    df["z_amt_vs_prov"] = (df["amount"] - df["prov_amt_median"]) / (df["prov_amt_p90"]-df["prov_amt_median"] + eps)
    df["z_amt_vs_cpt"]  = (df["amount"] - df["cpt_amt_median"]) / (df["cpt_amt_median"] + 5.0)
    df["ratio_units_vs_prov"] = (df["units"]+1)/(df["prov_units_median"]+1)
    df["ratio_units_vs_cpt"]  = (df["units"]+1)/(df["cpt_units_median"]+1)

    df["rare_pcpt"] = (df["pcpt_count"].fillna(0) < 3).astype(int)
    df["mod_59_or_25"] = df["modifier"].astype(str).isin(["59","25"]).astype(int)
    df["pos_office"] = (df["place_of_service"].astype(str)=="11").astype(int)
    df["pos_outpt"]  = (df["place_of_service"].astype(str)=="22").astype(int)

    feats = [
        "is_weekend","units","amount","prov_claims","cpt_count","member_day_claims",
        "z_amt_vs_prov","z_amt_vs_cpt","ratio_units_vs_prov","ratio_units_vs_cpt",
        "rare_pcpt","mod_59_or_25","pos_office","pos_outpt"
    ]
    X = df[feats].fillna(0).astype(float)
    return df, X

def heuristic_reasons(row):
    reasons=[]
    if row["z_amt_vs_prov"]>1.2: reasons.append("Amount far above provider norm")
    if row["ratio_units_vs_cpt"]>3: reasons.append("Unusually high units for CPT")
    if row["member_day_claims"]>3: reasons.append("Many same-day claims for member")
    if row["is_weekend"]==1 and row["pos_office"]==1: reasons.append("Office service on weekend")
    if row["mod_59_or_25"]==1 and row["ratio_units_vs_cpt"]>2: reasons.append("Modifier 25/59 with high units (possible unbundling)")
    if row["rare_pcpt"]==1: reasons.append("Rare provider–CPT combo")
    return reasons[:3]

def score_anomalies(X: pd.DataFrame):
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    model.fit(Xs)
    raw = -model.score_samples(Xs)            # higher = more anomalous
    # normalize 0..100
    rmin, rmax = float(raw.min()), float(raw.max())
    if rmax - rmin < 1e-12:
        return np.zeros_like(raw)
    return 100.0*(raw - rmin)/(rmax - rmin)

# ---------- UI ----------
file = st.file_uploader("Upload claims CSV", type=["csv"])

if file:
    try:
        with st.spinner("Scoring claims..."):
            df = pd.read_csv(file)
            # quick schema check
            required = {"claim_id","member_id","provider_id","service_date","cpt_code","dx_code","place_of_service","units","amount","modifier"}
            missing = required - set(map(str.lower, df.columns))
            if missing:
                st.error(f"Your CSV is missing columns: {', '.join(sorted(missing))}")
            else:
                # Normalize column names to lower to be forgiving
                df.columns = [c.lower() for c in df.columns]
                base, X = featurize(df)
                base["risk_score"] = score_anomalies(X)
                base["reasons"] = base.apply(heuristic_reasons, axis=1)
                out = base.sort_values("risk_score", ascending=False)

                st.success("Done. Showing top 300 by risk.")
                st.dataframe(out[["claim_id","provider_id","member_id","service_date","cpt_code",
                                  "units","amount","risk_score","reasons"]].head(300),
                             use_container_width=True)
                st.download_button(
                  with st.expander("CSV schema & tips"):
    st.markdown(
        """
- Dates: ISO format recommended (YYYY-MM-DD)  
- `place_of_service`: **11**=office, **22**=outpatient, **21**=inpatient  
- For best results, include realistic ranges for **amount** and **units**.  
        """
    )
