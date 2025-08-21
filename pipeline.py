import os, json
import pandas as pd, numpy as np
from dateutil.parser import isoparse
from dotenv import load_dotenv
load_dotenv()

# ---------------------------
# 1) Feature engineering
# ---------------------------
def _safe_dt(x):
    try: return isoparse(str(x)).date()
    except: return pd.NaT

def featurize(df: pd.DataFrame):
    df = df.copy()
    # types & basics
    df["service_date"] = df["service_date"].apply(_safe_dt)
    df["dow"] = df["service_date"].apply(lambda d: d.weekday() if pd.notna(d) else -1)
    df["is_weekend"] = (df["dow"].isin([5,6])).astype(int)
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    # provider-level stats
    grpP = df.groupby("provider_id").agg(
        prov_amt_median=("amount","median"),
        prov_amt_p90=("amount", lambda s: np.percentile(s,90) if len(s)>0 else 0),
        prov_units_median=("units","median"),
        prov_claims=("claim_id","count")
    )
    df = df.join(grpP, on="provider_id")

    # CPT-level stats
    grpC = df.groupby("cpt_code").agg(
        cpt_amt_median=("amount","median"),
        cpt_units_median=("units","median"),
        cpt_count=("claim_id","count")
    )
    df = df.join(grpC, on="cpt_code")

    # provider–CPT pair stats
    grpPC = df.groupby(["provider_id","cpt_code"]).agg(
        pcpt_amt_median=("amount","median"),
        pcpt_units_median=("units","median"),
        pcpt_count=("claim_id","count")
    )
    df = df.merge(grpPC, on=["provider_id","cpt_code"], how="left")

    # member/day density
    same_day = df.groupby(["member_id","service_date"]).size().rename("member_day_claims")
    df = df.join(same_day, on=["member_id","service_date"])
    df["member_day_claims"] = df["member_day_claims"].fillna(1)

    # z-scores & ratios (robust-ish)
    eps = 1e-6
    df["z_amt_vs_prov"] = (df["amount"] - df["prov_amt_median"]) / (df["prov_amt_p90"]-df["prov_amt_median"] + eps)
    df["z_amt_vs_cpt"]  = (df["amount"] - df["cpt_amt_median"]) / (df["cpt_amt_median"] + 5.0)
    df["ratio_units_vs_prov"] = (df["units"]+1)/(df["prov_units_median"]+1)
    df["ratio_units_vs_cpt"]  = (df["units"]+1)/(df["cpt_units_median"]+1)

    # rarity & modifiers
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
    return df, X, feats

# ---------------------------
# 2) Heuristic reasons
# ---------------------------
def heuristic_reasons(row):
    reasons=[]
    if row["z_amt_vs_prov"]>1.2: reasons.append("Amount far above provider norm")
    if row["ratio_units_vs_cpt"]>3: reasons.append("Unusually high units for CPT")
    if row["member_day_claims"]>3: reasons.append("High number of same-day claims for member")
    if row["is_weekend"]==1 and row["pos_office"]==1: reasons.append("Office service on weekend")
    if row["mod_59_or_25"]==1 and row["ratio_units_vs_cpt"]>2: reasons.append("Modifier 25/59 with high units (possible unbundling)")
    if row["rare_pcpt"]==1: reasons.append("Rare provider–CPT combo")
    return reasons[:3]

# ---------------------------
# 3) Cheap, pure-Python anomaly score (no sklearn)
#    We combine standardized features into one risk score
# ---------------------------
def _robust_z(series):
    s = series.astype(float)
    med = s.median()
    mad = (s - med).abs().median() + 1e-6
    return (s - med) / (1.4826*mad)

def simple_anomaly_score(df):
    # pick a few signals & robust-standardize them
    z1 = _robust_z(df["amount"])
    z2 = _robust_z(df["units"])
    z3 = df["z_amt_vs_prov"].fillna(0)
    z4 = df["z_amt_vs_cpt"].fillna(0)
    r1 = np.log1p(df["ratio_units_vs_cpt"]).fillna(0)
    r2 = np.log1p(df["member_day_claims"]).fillna(0)
    flags = (df["rare_pcpt"].fillna(0)*0.8 + df["mod_59_or_25"].fillna(0)*0.6 + df["is_weekend"].fillna(0)*0.4)

    # weighted sum
    raw = 0.9*z1 + 0.7*z2 + 0.8*z3 + 0.5*z4 + 0.6*r1 + 0.4*r2 + flags
    # min-max to 0..100
    raw = raw.replace([np.inf,-np.inf], np.nan).fillna(raw.median())
    mn, mx = float(raw.min()), float(raw.max())
    if mx - mn < 1e-9:
        return pd.Series(np.zeros(len(raw)))
    return 100.0*(raw - mn)/(mx - mn)

# ---------------------------
# 4) Optional LLM explanations
# ---------------------------
def llm_explain_openai(rows_json, model=os.getenv("OPENAI_MODEL","gpt-4o-mini")):
    from openai import OpenAI
    client = OpenAI()
    system = (
        "You are a healthcare claims auditor. "
        "Given claim rows with features (amount, units, z-scores, modifiers), "
        "propose the most likely denial/fraud category and a brief justification. "
        "Pick from: duplicate, upcoding, unbundling, not_medically_necessary, "
        "insufficient_documentation, place_of_service_mismatch, other. "
        "Output STRICT JSON list of objects: "
        '{"claim_id":..., "deny_reason":"...", "confidence":0-1, "rationale":"..."}'
    )
    user = "Analyze these rows:\n" + json.dumps(rows_json)[:12000]
    resp = client.responses.create(
        model=model,
        input=[{"role":"system","content":system},{"role":"user","content":user}],
        response_format={"type":"json_object"}
    )
    try:
        data = json.loads(resp.output[0].content[0].text)
        items = data.get("items", data if isinstance(data, list) else [])
        return items
    except Exception:
        return []

def llm_explain_anthropic(rows_json, model=os.getenv("ANTHROPIC_MODEL","claude-3-5-sonnet-20240620")):
    import anthropic, re, json as js
    client = anthropic.Anthropic()
    sys = (
        "You are a healthcare claims auditor. "
        "Return STRICT JSON as a list of objects "
        '[{"claim_id": "...","deny_reason":"...","confidence":0-1,"rationale":"..."}]. '
        "Reasons: duplicate, upcoding, unbundling, not_medically_necessary, "
        "insufficient_documentation, place_of_service_mismatch, other."
    )
    user = "Rows:\n" + json.dumps(rows_json)[:12000]
    msg = client.messages.create(model=model, max_tokens=800, system=sys, messages=[{"role":"user","content":user}])
    txt = "".join([b.text for b in msg.content if hasattr(b, "text")])
    m = re.search(r"\[.*\]", txt, flags=re.S)
    return js.loads(m.group(0)) if m else []

# ---------------------------
# 5) Public entrypoint
# ---------------------------
def score_claims(csv_path, use_llm=False, llm_provider="openai", top_k=50):
    df = pd.read_csv(csv_path)
    base_df, X, feats = featurize(df)

    # Pure-Python risk score
    base_df["risk_score"] = simple_anomaly_score(base_df)
    base_df["reasons"] = base_df.apply(heuristic_reasons, axis=1)

    # Optional LLM enrichment on top-K rows
    if use_llm and len(base_df)>0:
        top = (base_df.sort_values("risk_score", ascending=False)
                     .head(min(top_k, len(base_df))))
        cols = ["claim_id","provider_id","member_id","service_date","cpt_code","units","amount",
                "z_amt_vs_prov","z_amt_vs_cpt","ratio_units_vs_cpt","is_weekend","modifier","reasons"]
        rows = [{c: (r[c] if not (pd.isna(r[c])) else None) for c in cols} for _, r in top[cols].iterrows()]
        out = llm_explain_openai(rows) if llm_provider=="openai" else llm_explain_anthropic(rows)
        mapped = {o.get("claim_id"): o for o in out if isinstance(o, dict)}
        base_df["llm_deny_reason"] = base_df["claim_id"].map(lambda cid: mapped.get(cid,{}).get("deny_reason"))
        base_df["llm_confidence"]  = base_df["claim_id"].map(lambda cid: mapped.get(cid,{}).get("confidence"))
        base_df["llm_rationale"]   = base_df["claim_id"].map(lambda cid: mapped.get(cid,{}).get("rationale"))

    return base_df.sort_values("risk_score", ascending=False)


