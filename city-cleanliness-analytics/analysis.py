import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
R = {}  # results for the report

# ============ Module 1/2: collect + clean ============
raw = pd.read_csv("data/complaints_raw.csv")
prof = pd.read_csv("data/ward_profile.csv")
R["raw_rows"] = len(raw)
R["missing_resolution"] = int(raw.resolution_hours.isna().sum())
R["missing_timestamp"] = int(raw.timestamp.isna().sum())
R["duplicates"] = int(raw.duplicated(subset=["ward", "timestamp", "category", "resolution_hours", "channel"]).sum())
R["bad_ward_labels"] = int((~raw.ward.str.match(r"^Ward \d\d$")).sum())
R["category_variants_before"] = int(raw.category.nunique())
R["outliers_9999"] = int((raw.resolution_hours == 9999).sum())

df = raw.copy()
df["ward"] = df.ward.str.replace("ward-", "Ward ", regex=False)                      # standardise ward labels
df["category"] = (df.category.str.strip().str.replace("_", " ").str.lower().str.capitalize())  # unify labels
df["category"] = df.category.replace({"Littering / dumping": "Littering / dumping"})
df = df.drop_duplicates(subset=["ward", "timestamp", "category", "resolution_hours", "channel"])
df = df.dropna(subset=["timestamp"])
df.loc[df.resolution_hours > 240, "resolution_hours"] = np.nan                        # impossible values -> missing
R["imputed"] = int(df.resolution_hours.isna().sum())
med = df.groupby("category").resolution_hours.transform("median")
df["resolution_hours"] = df.resolution_hours.fillna(med)                              # impute by category median
df["timestamp"] = pd.to_datetime(df.timestamp)
df["month"] = df.timestamp.dt.month
df["hour"] = df.timestamp.dt.hour
df["dow"] = df.timestamp.dt.dayofweek
assert df.isna().sum().sum() == 0
R["clean_rows"] = len(df)
R["category_variants_after"] = int(df.category.nunique())
df.to_csv("data/complaints_clean.csv", index=False)

# ============ ward-level integration ============
g = df.groupby("ward")
agg = pd.DataFrame({
    "complaints": g.size(),
    "avg_resolution_hrs": g.resolution_hours.mean(),
    "resolved_within_48h_pct": g.resolution_hours.apply(lambda s: (s <= 48).mean() * 100),
}).reset_index()
W = prof.merge(agg, on="ward")
W["complaints_per_10k_month"] = W.complaints / (W.population_k / 10) / 12
W = W.round(2)
W.to_csv("data/ward_master.csv", index=False)
R["wards"] = len(W)

# ============ Module 3: EDA ============
ind = {
    "Complaints per 10k residents (monthly)": "complaints_per_10k_month",
    "Avg resolution time (hours)": "avg_resolution_hrs",
    "Resolved within 48 h (%)": "resolved_within_48h_pct",
    "Waste pickups per week": "pickups_per_week",
    "Doorstep collection coverage (%)": "doorstep_coverage_pct",
    "Public toilets functional (%)": "toilet_functional_pct",
    "Inspection score (0-100)": "inspection_score",
    "Bins per sq km": "bins_per_sqkm",
}
desc = []
for lab, c in ind.items():
    s = W[c]
    desc.append([lab, round(s.mean(), 1), round(s.std(ddof=1), 1), round(s.min(), 1), round(s.max(), 1)])
R["table1"] = desc

cols = list(ind.values())
z = (W[cols] - W[cols].mean()) / W[cols].std(ddof=1)
R["z_flags"] = [(W.ward[i], cols[j], round(float(z.iloc[i, j]), 2)) for i in range(len(W)) for j in range(len(cols)) if abs(z.iloc[i, j]) > 1.8]
q1, q3 = W.complaints_per_10k_month.quantile([.25, .75]); fence = q3 + 1.5 * (q3 - q1)
R["iqr_fence"] = round(float(fence), 2)
R["iqr_outliers"] = W.loc[W.complaints_per_10k_month > fence, "ward"].tolist()

cm = W[["complaints_per_10k_month", "avg_resolution_hrs", "pickups_per_week", "doorstep_coverage_pct",
        "inspection_score", "toilet_functional_pct"]].corr()
R["corr"] = {
    "pickups_vs_complaints": round(cm.loc["pickups_per_week", "complaints_per_10k_month"], 2),
    "inspection_vs_complaints": round(cm.loc["inspection_score", "complaints_per_10k_month"], 2),
    "resolution_vs_inspection": round(cm.loc["avg_resolution_hrs", "inspection_score"], 2),
    "coverage_vs_inspection": round(cm.loc["doorstep_coverage_pct", "inspection_score"], 2),
    "complaints_vs_resolution": round(cm.loc["complaints_per_10k_month", "avg_resolution_hrs"], 2),
}

# ============ Module 5: Cleanliness Index ============
def mm(s, inv=False):
    v = (s - s.min()) / (s.max() - s.min()) * 100
    return 100 - v if inv else v
weights = {"complaints": .25, "resolution": .20, "pickups": .20, "inspection": .20, "coverage": .15}
W["s_complaints"] = mm(W.complaints_per_10k_month, inv=True)
W["s_resolution"] = mm(W.avg_resolution_hrs, inv=True)
W["s_pickups"] = mm(W.pickups_per_week)
W["s_inspection"] = mm(W.inspection_score)
W["s_coverage"] = mm(W.doorstep_coverage_pct)
W["index"] = (weights["complaints"] * W.s_complaints + weights["resolution"] * W.s_resolution +
              weights["pickups"] * W.s_pickups + weights["inspection"] * W.s_inspection +
              weights["coverage"] * W.s_coverage).round(1)
rank = W.sort_values("index", ascending=False)[["ward", "zone", "index"]].reset_index(drop=True)
R["rank_top3"] = rank.head(3).values.tolist()
R["rank_bottom3"] = rank.tail(3).values.tolist()
R["rank_all"] = rank.values.tolist()
W.to_csv("data/ward_index.csv", index=False)

# PCA (numpy SVD)
zc = ((W[cols] - W[cols].mean()) / W[cols].std(ddof=1)).values
U, sv, Vt = np.linalg.svd(zc, full_matrices=False)
ev = sv ** 2 / (sv ** 2).sum()
pc1 = zc @ Vt[0]
r_pc = abs(np.corrcoef(pc1, W["index"])[0, 1])
R["pca"] = {"pc1": round(float(ev[0]) * 100, 1), "pc12": round(float(ev[:2].sum()) * 100, 1), "r_index": round(float(r_pc), 2)}

# sensitivity: equal weights vs chosen weights
eq = W[["s_complaints", "s_resolution", "s_pickups", "s_inspection", "s_coverage"]].mean(axis=1)
R["rank_corr_equal_weights"] = round(float(pd.Series(W["index"]).rank().corr(eq.rank())), 3)

# hotspots / patterns
mon = df.groupby("month").size()
R["monthly"] = mon.tolist()
R["peak_month"] = int(mon.idxmax()); R["low_month"] = int(mon.idxmin())
R["monsoon_rise_pct"] = round(float((mon[[7, 8, 9]].mean() / mon[[1, 2, 3]].mean() - 1) * 100), 1)
cat = df.category.value_counts()
R["cat_share"] = {k: round(v / len(df) * 100, 1) for k, v in cat.items()}
hr = df.groupby("hour").size()
R["peak_hours"] = [int(h) for h in hr.sort_values(ascending=False).index[:3]]
R["morning_share"] = round(float(df[df.hour.between(6, 11)].shape[0] / len(df) * 100), 1)
R["resolved_48h_all"] = round(float((df.resolution_hours <= 48).mean() * 100), 1)
R["avg_res_all"] = round(float(df.resolution_hours.mean()), 1)
R["channel_share"] = {k: round(v / len(df) * 100, 1) for k, v in df.channel.value_counts().items()}
top_w = W.sort_values("complaints_per_10k_month", ascending=False).head(3)[["ward", "complaints_per_10k_month"]].values.tolist()
R["top_complaint_wards"] = top_w
R["total_complaints"] = int(len(df))

# ============ Module 4: figures ============
C1, C2, C3 = "#2E7D5B", "#C0504D", "#4F6D8F"
# Fig 1: data overview (sources/records + cleaning effect)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
steps = ["Raw rows", "After removing\nduplicates", "After dropping\nmissing time"]
n0 = len(raw); n1 = n0 - R["duplicates"]; n2 = len(df)
ax[0].bar(steps, [n0, n1, n2], color=[C3, C3, C1]); ax[0].set_title("(a) Record count through cleaning")
for i, v in enumerate([n0, n1, n2]): ax[0].text(i, v + 40, f"{v:,}", ha="center")
ax[0].set_ylim(0, n0 * 1.15)
cs = cat.sort_values()
ax[1].barh(cs.index, cs.values, color=C1); ax[1].set_title("(b) Complaints by category (cleaned)")
plt.tight_layout(); plt.savefig("figs/fig1.png", dpi=200); plt.close()

# Fig 2: monthly trend + hour of day
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
mn = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
ax[0].plot(mn, mon.values, marker="o", color=C2); ax[0].fill_between(range(12), mon.values, alpha=.15, color=C2)
ax[0].set_title("(a) Complaints per month, 2025"); ax[0].set_ylabel("Complaints")
ax[1].bar(hr.index, hr.values, color=C3); ax[1].set_title("(b) Complaints by hour of day"); ax[1].set_xlabel("Hour")
plt.tight_layout(); plt.savefig("figs/fig2.png", dpi=200); plt.close()

# Fig 3: ward ranking + correlation heatmap
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 1.05]})
r2 = W.sort_values("index")
colors = [C2 if v < 40 else (C1 if v >= 60 else "#B8A24A") for v in r2["index"]]
ax[0].barh([w.replace("Ward ", "W") for w in r2.ward], r2["index"], color=colors)
ax[0].set_title("(a) Ward ranking by Cleanliness Index"); ax[0].set_xlim(0, 105)
for i, v in enumerate(r2["index"]): ax[0].text(v + 1, i, f"{v}", va="center", fontsize=6.5)
labs = ["Compl./10k", "Resol. hrs", "Pickups/wk", "Doorstep %", "Inspection", "Toilets %"]
im = ax[1].imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax[1].set_xticks(range(6)); ax[1].set_xticklabels(labs, rotation=45, ha="right", fontsize=6.5)
ax[1].set_yticks(range(6)); ax[1].set_yticklabels(labs, fontsize=6.5)
for i in range(6):
    for j in range(6): ax[1].text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color=("white" if abs(cm.values[i, j]) > .7 else "black"))
ax[1].set_title("(b) Pearson correlation matrix")
plt.tight_layout(); plt.savefig("figs/fig3.png", dpi=200); plt.close()

# Fig 4: ward x month heatmap + hour x weekday heatmap
piv = df.pivot_table(index="ward", columns="month", values="complaint_id", aggfunc="count", fill_value=0)
piv = piv.div(W.set_index("ward").population_k / 10, axis=0).loc[rank.ward]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
im = ax[0].imshow(piv.values, cmap="YlOrRd", aspect="auto")
ax[0].set_yticks(range(15)); ax[0].set_yticklabels([w.replace("Ward ", "W") for w in piv.index], fontsize=6)
ax[0].set_xticks(range(12)); ax[0].set_xticklabels(mn, fontsize=6.5)
ax[0].set_title("(a) Complaints per 10k residents\n(wards sorted best to worst)", fontsize=7.5)
plt.colorbar(im, ax=ax[0], fraction=.045)
hw = df.pivot_table(index="dow", columns="hour", values="complaint_id", aggfunc="count", fill_value=0)
im2 = ax[1].imshow(hw.values, cmap="Blues", aspect="auto")
ax[1].set_yticks(range(7)); ax[1].set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=6.5)
ax[1].set_xticks(range(0, 24, 3)); ax[1].set_xticklabels(range(0, 24, 3), fontsize=6.5)
ax[1].set_title("(b) Complaints by weekday and hour", fontsize=7.5); plt.colorbar(im2, ax=ax[1], fraction=.045)
plt.tight_layout(); plt.savefig("figs/fig4.png", dpi=200); plt.close()

# Fig 5: index components for best 3 / worst 3 + what-if
comp = ["s_complaints", "s_resolution", "s_pickups", "s_inspection", "s_coverage"]
cl = ["Complaints", "Resolution", "Pickups", "Inspection", "Coverage"]
sel = pd.concat([W.sort_values("index", ascending=False).head(3), W.sort_values("index").head(3)])
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
x = np.arange(5); wd = .13
for k, (_, row) in enumerate(sel.iterrows()):
    ax[0].bar(x + (k - 2.5) * wd, [row[c] for c in comp], wd, label=row.ward.replace("Ward ", "W"),
              color=(C1 if k < 3 else C2), alpha=(1 - .22 * (k % 3)))
ax[0].set_xticks(x); ax[0].set_xticklabels(cl, fontsize=6.5); ax[0].set_title("(a) Components: top 3 (green) vs bottom 3 (red)", fontsize=7.5)
ax[0].set_ylim(0, 128); ax[0].legend(fontsize=5.5, ncol=6, loc='upper center')

def simulate(wname, pick_up_add, res_cut_pct, cov_add):
    r = W[W.ward == wname].iloc[0]
    p = min(W.pickups_per_week.max(), r.pickups_per_week + pick_up_add)
    rs = r.avg_resolution_hrs * (1 - res_cut_pct / 100)
    cv = min(99, r.doorstep_coverage_pct + cov_add)
    sp = (p - W.pickups_per_week.min()) / (W.pickups_per_week.max() - W.pickups_per_week.min()) * 100
    sr = 100 - (rs - W.avg_resolution_hrs.min()) / (W.avg_resolution_hrs.max() - W.avg_resolution_hrs.min()) * 100
    sc = (cv - W.doorstep_coverage_pct.min()) / (W.doorstep_coverage_pct.max() - W.doorstep_coverage_pct.min()) * 100
    sr = min(100, sr); sp = min(100, sp); sc = min(100, sc)
    new = (weights["complaints"] * r.s_complaints + weights["resolution"] * sr + weights["pickups"] * sp +
           weights["inspection"] * r.s_inspection + weights["coverage"] * sc)
    return round(float(r["index"]), 1), round(float(new), 1)
worst = rank.ward.iloc[-1]
b, a = simulate(worst, 1.5, 25, 8)
R["sim"] = {"ward": worst, "before": b, "after": a, "pickup_add": 1.5, "res_cut": 25, "cov_add": 8}
ax[1].bar(["Before", "After\nintervention"], [b, a], color=[C2, C1], width=.5)
for i, v in enumerate([b, a]): ax[1].text(i, v + 1, str(v), ha="center")
ax[1].set_ylim(0, 100); ax[1].set_title(f"(b) What-if for {worst}: +1.5 pickups/wk,\n-25% resolution time, +8 pts coverage", fontsize=7.5)
plt.tight_layout(); plt.savefig("figs/fig5.png", dpi=200); plt.close()

# Fig 6: architecture
fig, ax = plt.subplots(figsize=(7.2, 2.8)); ax.axis("off")
def box(x, y, w, h, t):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc="#E9EEF2", ec="#333", lw=.8)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.5)
box(.02, .78, .96, .16, "Users: Municipal officer  |  Ward supervisor  |  Policy analyst  |  Citizen (view only)")
box(.02, .50, .30, .20, "DATA LAYER\ncomplaints_raw.csv\nward_profile.csv")
box(.35, .50, .30, .20, "PRE-PROCESSING LAYER\nclean labels, drop duplicates,\nimpute, integrate to ward master")
box(.68, .50, .30, .20, "ANALYSIS LAYER\nPandas statistics, z-score / IQR,\ncorrelation, PCA")
box(.02, .16, .30, .20, "SCORING LAYER\nCleanliness Index\n(weighted, 0-100)")
box(.35, .16, .30, .20, "SIMULATION LAYER\nwhat-if on pickups,\nresolution time, coverage")
box(.68, .16, .30, .20, "VISUALISATION LAYER\nMatplotlib charts,\nheatmaps, ranking")
for x in (.17, .50, .83): ax.annotate("", xy=(x, .70), xytext=(x, .78), arrowprops=dict(arrowstyle="->"))
for x in (.17, .50, .83): ax.annotate("", xy=(x, .36), xytext=(x, .50), arrowprops=dict(arrowstyle="->"))
plt.savefig("figs/fig6.png", dpi=200, bbox_inches="tight"); plt.close()

json.dump(R, open("results.json", "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
