import os, json
import pandas as pd, numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from dotenv import load_dotenv

load_dotenv()

# ---------- Feature engineering ----------
def featurize(df: pd.DataFrame):
    df = df.copy()

    # types & basics
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
    df["dow"] = df["service_date"].dt.weekday.fillna(-1).astype(int)
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0).astype(float)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).astype(float)

    # provider-level stats
    grpP = df.groupby("provider_id").agg(
        prov_amt_median=("amount","median"),
        prov_amt_p90=("amount", lambda s: np.percentile(s,90) if len(s)>0 else 0),
        prov_units_median=("units","median"),
        prov_claims=("claim_id","count")
    )
    df = df.join(grpP, on="provider_id")

    # CPT-level stats (global)
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

    # derived features
    eps = 1e-6
    df["z_amt_vs_prov"] = (df["amount"] - df["prov_amt_median"]) / (df["prov_amt_p90"] - df["prov_amt_median"] + eps)
    df["z_amt_vs_cpt"]  = (df["amount"] - df["cpt_amt_median"]) / (df["cpt_amt_median"] + 5.0 + eps)
    df["ratio_units_vs_prov"] = (df["units"] + 1) / (df["prov_units_median"] + 1)
    df["ratio_units_vs_cpt"]  = (df["units"] + 1) / (df["cpt_units_median"] + 1)

    df["rare_pcpt"] = (df["pcpt_count"].fillna(0) < 3).astype(int)
    df["mod_59_or_25"] = df["modifier"].astype(str).isin(["59","25"]).astype(int)
    df["pos_office"] = (df["place_of_service"].astype(str) == "11").astype(int)
    df["pos_outpt"]  = (df["place_of_service"].astype(str) == "22").astype(int)

    feats = [
        "is_weekend","units","amount","prov_claims","cpt_count","member_day_claims",
        "z_amt_vs_prov","z_amt_vs_cpt","ratio_units_vs_prov","ratio_units_vs_cpt",
        "rare_pcpt","mod_59_or_25","pos_office","pos_outpt"
    ]
    X = df[feats].fillna(0).astype(float)
    return df, X, feats

# ---------- Anomaly model ----------
class AnomalyScorer:
    def __init__(self):
        self.scaler = RobustScaler()
        self.model  = IsolationForest(
            n_estimators=200, contamination="auto", random_state=42, n_jobs=-1
        )

    def fit(self, X):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)

    def score(self, X):
        Xs = self.scaler.transform(X)
        raw = -self.model.score_samples(Xs)  # higher => more anomalous
        rmin, rmax = float(raw.min()), float(raw.max())
        if rmax - rmin < 1e-9:
            return np.zeros_like(raw)
        return 100.0 * (raw - rmin) / (rmax - rmin)

# ---------- Heuristics ----------
def heuristic_reasons(row):
    reasons = []
    if row["z_amt_vs_prov"] > 1.2:
        reasons.append("Amount far above provider norm")
    if row["ratio_units_vs_cpt"] > 3:
        reasons.append("Unusually high units for CPT")
    if row["member_day_claims"] > 3:
        reasons.append("High number of same-day claims for member")
    if row["is_weekend"] == 1 and row["pos_office"] == 1:
        reasons.append("Office service on weekend")
    if row["mod_59_or_25"] == 1 and row["ratio_units_vs_cpt"] > 2:
        reasons.append("Modifier 25/59 with high units (possible unbundling)")
    if row["rare_pcpt"] == 1:
        reasons.append("Rare provider–CPT combo")
    return reasons[:3]

# ---------- Optional LLM explanations ----------
def llm_explain_openai(rows_json, model=os.getenv("OPENAI_MODEL","gpt-4o-mini")):
    try:
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
        data = json.loads(resp.output[0].content[0].text)
        return data.get("items", data if isinstance(data, list) else [])
    except Exception:
        return []

def llm_explain_anthropic(rows_json, model=os.getenv("ANTHROPIC_MODEL","claude-3-5-sonnet-20240620")):
    try:
        import anthropic, re, json as js
        client = anthropic.Anthropic()
        sys = (
            "You are a healthcare claims auditor. "
            "Return STRICT JSON as a list of objects "
            '[{"claim_id":"...","deny_reason":"...","confidence":0-1,"rationale":"..."}]. '
            "Reasons: duplicate, upcoding, unbundling, not_medically_necessary, "
            "insufficient_documentation, place_of_service_mismatch, other."
        )
        user = "Rows:\n" + json.dumps(rows_json)[:12000]
        msg = client.messages.create(model=model, max_tokens=800, system=sys, messages=[{"role":"user","content":user}])
        txt = "".join([b.text for b in msg.content if hasattr(b, "text")])
        m = re.search(r"\[.*\]", txt, flags=re.S)
        return js.loads(m.group(0)) if m else []
    except Exception:
        return []

# ---------- Public entrypoint ----------
def score_claims(csv_path, use_llm=False, llm_provider="openai", top_k=50):
    df = pd.read_csv(csv_path)
    base_df, X, feats = featurize(df)

    scorer = AnomalyScorer()
    scorer.fit(X)
    base_df["risk_score"] = scorer.score(X)
    base_df["reasons"] = base_df.apply(heuristic_reasons, axis=1)

    if use_llm and len(base_df) > 0:
        top = base_df.sort_values("risk_score", ascending=False).head(min(top_k, len(base_df)))
        cols = ["claim_id","provider_id","member_id","service_date","cpt_code","units","amount",
                "z_amt_vs_prov","z_amt_vs_cpt","ratio_units_vs_cpt","is_weekend","modifier","reasons"]
        # safe JSON payload
        top_small = top[cols].copy()
        top_small = top_small.where(pd.notna(top_small), None)
        if "reasons" in top_small.columns:
            top_small["reasons"] = top_small["reasons"].apply(
                lambda x: x if isinstance(x, list) else ([] if x is None else [str(x)])
            )
        rows = top_small.to_dict(orient="records")

        if llm_provider == "openai":
            out = llm_explain_openai(rows)
        else:
            out = llm_explain_anthropic(rows)

        mapped = {o.get("claim_id"): o for o in out if isinstance(o, dict)}
        base_df["llm_deny_reason"] = base_df["claim_id"].map(lambda cid: mapped.get(cid, {}).get("deny_reason"))
        base_df["llm_confidence"]  = base_df["claim_id"].map(lambda cid: mapped.get(cid, {}).get("confidence"))
        base_df["llm_rationale"]   = base_df["claim_id"].map(lambda cid: mapped.get(cid, {}).get("rationale"))

    return base_df.sort_values("risk_score", ascending=False)


