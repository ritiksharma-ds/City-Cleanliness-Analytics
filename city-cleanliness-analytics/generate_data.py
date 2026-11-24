"""Generates a SIMULATED civic cleanliness dataset for 15 wards (calendar year 2025).
The data is synthetic (seeded) and deliberately messy so that Module 2 (pre-processing) has real work to do."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
wards = [f"Ward {i:02d}" for i in range(1, 16)]
zones = ["North", "North", "North", "South", "South", "South", "East", "East", "East",
         "West", "West", "West", "Central", "Central", "Central"]
# latent "true cleanliness" per ward drives every indicator, so correlations are realistic
latent = np.clip(rng.normal(0, 1, 15), -1.8, 1.8)
population_k = rng.integers(55, 160, 15)
area = rng.uniform(4, 16, 15).round(1)

ward = pd.DataFrame({
    "ward": wards, "zone": zones, "population_k": population_k, "area_sqkm": area,
    "pickups_per_week": np.clip(np.round(5.2 + 1.1 * latent + rng.normal(0, .8, 15), 1), 2.5, 7),
    "doorstep_coverage_pct": np.clip(np.round(82 + 8 * latent + rng.normal(0, 5.5, 15), 1), 55, 99),
    "inspection_score": np.clip(np.round(68 + 9 * latent + rng.normal(0, 6, 15), 1), 40, 96),
    "toilet_functional_pct": np.clip(np.round(80 + 8 * latent + rng.normal(0, 7, 15), 1), 50, 99),
    "bins_per_sqkm": np.clip(np.round(14 + 4 * latent + rng.normal(0, 3.5, 15), 1), 4, 28),
})
ward.to_csv("data/ward_profile.csv", index=False)

# complaint-level records
cats = ["Garbage overflow", "Missed collection", "Littering / dumping", "Drain blockage",
        "Dead animal", "Public toilet issue", "Street sweeping"]
cat_p = np.array([.27, .21, .16, .12, .05, .09, .10])
month_w = np.array([.75, .75, .80, .90, 1.0, 1.15, 1.45, 1.50, 1.30, 1.00, .85, .80])  # monsoon bump
rows = []
for k, w in enumerate(wards):
    rate = (2.4 - 0.55 * latent[k]) * np.exp(rng.normal(0, .18)) * population_k[k] / 10  # base monthly complaints
    for m in range(12):
        n = rng.poisson(rate * month_w[m] * 1.0)
        days = rng.integers(1, 29, n)
        hour_p = np.array([1,1,1,1,1,2,4,8,9,8,6,5,5,5,5,6,7,8,8,6,4,3,2,1], float); hour_p /= hour_p.sum()
        hours = rng.choice(24, n, p=hour_p)
        cat = rng.choice(cats, n, p=cat_p)
        res_h = np.clip(rng.lognormal(np.log(max(14, 30 - 6 * latent[k] + rng.normal(0, 5))), .55, n), 2, 240).round(1)
        for d, h, c, r in zip(days, hours, cat, res_h):
            rows.append((w, f"2025-{m+1:02d}-{d:02d} {h:02d}:{rng.integers(0,60):02d}", c, r,
                         rng.choice(["App", "Helpline", "Web"], p=[.62, .25, .13])))
df = pd.DataFrame(rows, columns=["ward", "timestamp", "category", "resolution_hours", "channel"])
# ---- inject realistic mess ----
n = len(df)
messy = df.copy()
idx = rng.choice(n, int(.06 * n), replace=False)
messy.loc[idx, "category"] = messy.loc[idx, "category"].str.upper()
idx = rng.choice(n, int(.04 * n), replace=False)
messy.loc[idx, "category"] = messy.loc[idx, "category"].str.replace(" ", "_") + " "
idx = rng.choice(n, int(.03 * n), replace=False); messy.loc[idx, "ward"] = messy.loc[idx, "ward"].str.replace("Ward ", "ward-")
idx = rng.choice(n, int(.02 * n), replace=False); messy.loc[idx, "resolution_hours"] = np.nan
idx = rng.choice(n, int(.01 * n), replace=False); messy.loc[idx, "timestamp"] = np.nan
idx = rng.choice(n, 40, replace=False); messy.loc[idx, "resolution_hours"] = 9999   # data-entry errors
dups = messy.sample(int(.015 * n), random_state=1)
messy = pd.concat([messy, dups]).sample(frac=1, random_state=7).reset_index(drop=True)
messy.insert(0, "complaint_id", [f"C{100000+i}" for i in range(len(messy))])
messy.to_csv("data/complaints_raw.csv", index=False)
print("raw rows", len(messy), "clean base rows", n)
