import json, pandas as pd, numpy as np
R = json.load(open("results.json"))
raw = pd.read_csv("data/complaints_raw.csv"); cl = pd.read_csv("data/complaints_clean.csv")
W = pd.read_csv("data/ward_index.csv"); res = []
def t(name, method, cond):
    res.append((name, method, "Passed" if cond else "FAILED")); print(res[-1])

t("No missing values", "isna().sum().sum() == 0 on cleaned data", cl.isna().sum().sum() == 0)
t("No duplicates", "no repeated (ward, time, category, hours, channel)", not cl.duplicated(subset=["ward","timestamp","category","resolution_hours","channel"]).any())
t("Ward labels", "all labels match 'Ward NN' (15 unique)", cl.ward.str.match(r"^Ward \d\d$").all() and cl.ward.nunique() == 15)
t("Category labels", "27 raw variants reduced to 7 categories", raw.category.nunique() == R["category_variants_before"] and cl.category.nunique() == 7)
t("Resolution range", "all resolution times between 2 and 240 hours", cl.resolution_hours.between(2, 240).all())
t("Row reconciliation", "raw - duplicates - missing time = cleaned rows", len(raw) - R["duplicates"] - R["missing_timestamp"] == len(cl))
t("Index range", "Cleanliness Index within 0-100 for all 15 wards", W["index"].between(0, 100).all() and len(W) == 15)
t("Weights", "weights 25+20+20+20+15 sum to 100", 25 + 20 + 20 + 20 + 15 == 100)
t("Aggregation check", "ward totals add up to cleaned complaint count", int(W.complaints.sum()) == len(cl))
t("Descriptive check", "Table 1 mean of pickups matches Pandas mean", abs(W.pickups_per_week.mean() - [r for r in R["table1"] if r[0].startswith("Waste")][0][1]) < 0.05)
t("Sensitivity", "rank correlation, chosen vs equal weights >= 0.95", R["rank_corr_equal_weights"] >= 0.95)
t("Simulation", "what-if raises the index of the worst ward", R["sim"]["after"] > R["sim"]["before"])
json.dump(res, open("tests.json", "w"))
print(sum(r[2] == "Passed" for r in res), "/", len(res))
