import pandas as pd, numpy as np, random, datetime as dt

rng = np.random.default_rng(7)
CPT = ["99213","99214","93000","80050","97110","97012","36415","J1885"]
DX  = ["E11.9","I10","M54.5","J06.9","K21.9"]
prov = [f"P{n:03d}" for n in range(1,21)]
mem  = [f"M{n:05d}" for n in range(1,801)]
rows=[]
start=dt.date(2025,1,1)

for i in range(4000):
    p = random.choice(prov)
    m = random.choice(mem)
    d = start + dt.timedelta(days=rng.integers(0,120))
    cpt = random.choice(CPT)
    units = max(1,int(abs(rng.normal(1.5,0.8))))
    base = {"99213":95,"99214":140,"93000":55,"80050":110,"97110":45,"97012":38,"36415":10,"J1885":25}[cpt]
    amt = round(max(10, rng.normal(base*units, base*0.25)),2)
    mod = random.choice(["","25","59","76","80"])
    pos = random.choice(["11","22","21"])  # office, outpatient, inpatient
    dx  = random.choice(DX)
    rows.append([f"C{i:06d}",m,p,d.isoformat(),cpt,dx,pos,units,amt,mod])

# inject anomalies
for j in range(40):
    i = rng.integers(0,len(rows))
    rows[i][7] = int(rows[i][7]*rng.integers(6,12))
for j in range(35):
    i = rng.integers(0,len(rows))
    rows[i][8] = round(rows[i][8]*rng.uniform(4.5,9.0),2)

pd.DataFrame(rows, columns=["claim_id","member_id","provider_id","service_date","cpt_code",
                            "dx_code","place_of_service","units","amount","modifier"]
            ).to_csv("sample_claims.csv", index=False)
print("Wrote sample_claims.csv")
