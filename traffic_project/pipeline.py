"""Urban Traffic Management: Pattern Mining and Congestion Prediction.
Run:  python pipeline.py [--csv your_traffic.csv]
Outputs: data/*.csv, data/*.arff (Weka/Orange), results.json, dashboard.html
"""
import json, sys, itertools, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, silhouette_score,
                             r2_score, mean_absolute_error, mean_squared_error)
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
rng = np.random.default_rng(42)
LEVELS = ["Low", "Moderate", "High", "Severe"]
VTYPES = ["Car", "Two-wheeler", "Bus", "Truck", "Auto-rickshaw"]

# ---------- 1. Data (synthetic generator; replace with --csv for real data) ----------
AREAS = {  # capacity (veh/h), peak hours, vehicle mix
    "Downtown CBD":     (2600, [9, 18],  [.45, .25, .08, .04, .18]),
    "Tech Park":        (2200, [9, 18],  [.55, .30, .10, .01, .04]),
    "Railway Station":  (2400, [8, 19],  [.25, .25, .15, .05, .30]),
    "Airport Road":     (2000, [7, 21],  [.60, .15, .10, .10, .05]),
    "University Zone":  (1700, [9, 17],  [.25, .50, .12, .01, .12]),
    "Old Market":       (1900, [12, 20], [.20, .35, .04, .06, .35]),
    "Industrial Area":  (1800, [7, 16],  [.20, .20, .10, .45, .05]),
    "Residential North": (1500, [8, 19], [.40, .38, .05, .02, .15]),
}

def generate(days=120):
    rows, start = [], pd.Timestamp("2026-06-01")
    for a, (cap, peaks, mix) in AREAS.items():
        for ts in pd.date_range(start, periods=days * 24, freq="h"):
            h, wk = ts.hour, ts.dayofweek >= 5
            prof = 0.12 + sum(np.exp(-((h - p) ** 2) / (2 * 1.9 ** 2)) for p in peaks) * (0.55 if wk else 0.82)
            if wk: prof += 0.12 * np.exp(-((h - 15) ** 2) / 18)
            rain = float(rng.random() < 0.12) * rng.uniform(1, 12)
            event = int(rng.random() < 0.015)
            vol = cap * prof * (1 + 0.04 * rain / 6) * (1 + 0.25 * event) * rng.normal(1, 0.07)
            idx = np.clip(vol / cap * 100 * (1 + 0.012 * rain) + rng.normal(0, 3), 0, 100)
            share = rng.dirichlet(np.array(mix) * 200)
            rows.append([ts, a, h, ts.dayofweek, int(wk), round(rain, 1), event, int(vol),
                         *(np.round(share * vol).astype(int)), round(60 * (1 - 0.72 * idx / 100) + rng.normal(0, 1.5), 1),
                         round(idx, 1)])
    cols = ["timestamp", "area", "hour", "dow", "weekend", "rain_mm", "event", "volume", *VTYPES, "avg_speed", "congestion_idx"]
    return pd.DataFrame(rows, columns=cols)

csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
df = pd.read_csv(csv, parse_dates=["timestamp"]) if csv else generate()

# ---------- 2. Preprocessing ----------
df = df.sort_values(["area", "timestamp"]).drop_duplicates(["area", "timestamp"]).reset_index(drop=True)
num = df.select_dtypes("number").columns
df[num] = df.groupby("area")[num].transform(lambda s: s.interpolate().bfill().ffill())  # missing values
for c in ["volume", "avg_speed", "congestion_idx"]:                                       # outliers (IQR clip)
    q1, q3 = df[c].quantile([.25, .75]); df[c] = df[c].clip(q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1))
df["level"] = pd.cut(df.congestion_idx, [-1, 35, 60, 80, 101], labels=LEVELS).astype(str)
g = df.groupby("area").congestion_idx
df["lag_24"], df["lag_168"] = g.shift(24), g.shift(168)
df = df.dropna().reset_index(drop=True)
df["area_code"] = df.area.astype("category").cat.codes
area_names = list(df.area.astype("category").cat.categories)
df.to_csv(ROOT / "data/traffic_clean.csv", index=False)
with open(ROOT / "data/traffic.arff", "w") as f:  # Weka / Orange input
    f.write("@relation traffic\n@attribute area {%s}\n@attribute hour numeric\n@attribute dow numeric\n"
            "@attribute weekend numeric\n@attribute rain_mm numeric\n@attribute event numeric\n"
            "@attribute lag_24 numeric\n@attribute lag_168 numeric\n@attribute level {%s}\n@data\n"
            % (",".join(f"'{a}'" for a in area_names), ",".join(LEVELS)))
    for r in df.itertuples():
        f.write(f"'{r.area}',{r.hour},{r.dow},{r.weekend},{r.rain_mm},{r.event},{r.lag_24},{r.lag_168},{r.level}\n")

# ---------- 3. Pattern mining (Apriori, support/confidence/lift) ----------
def item_frame(d):
    t = pd.DataFrame({"area": "Area=" + d.area,
                      "slot": "Time=" + pd.cut(d.hour, [-1, 5, 10, 15, 20, 23], labels=["Night", "MorningPeak", "Midday", "EveningPeak", "Late"]).astype(str),
                      "day": np.where(d.weekend == 1, "Day=Weekend", "Day=Weekday"),
                      "rain": np.where(d.rain_mm > 0, "Rain=Yes", "Rain=No"),
                      "level": "Level=" + d.level})
    return t
T = item_frame(df); n = len(T)
tx = [frozenset(r) for r in T.values]
def support(items): return sum(items <= t for t in tx) / n
freq1 = {frozenset([i]): support(frozenset([i])) for i in set(itertools.chain(*tx))}
freq = {k: v for k, v in freq1.items() if v >= 0.01}
rules, cur = [], list(freq)
for size in (2, 3):
    cand = {a | b for a, b in itertools.combinations(cur, 2) if len(a | b) == size}
    cur = [c for c in cand if support(c) >= 0.01]
    for c in cur:
        s = support(c)
        for it in c:
            if it.startswith("Level="):
                ante = c - {it}; conf = s / support(ante); lift = conf / freq1[frozenset([it])]
                if it in ("Level=Severe", "Level=High") and conf >= 0.4 and lift > 1.3:
                    rules.append({"if": " + ".join(sorted("Rain" if x=="Rain=Yes" else x.split("=")[1] for x in ante if x!="Rain=No")), "then": it.split("=")[1],
                                  "support": round(s, 3), "confidence": round(conf, 2), "lift": round(lift, 2)})
rules = sorted(rules, key=lambda r: (-r["lift"], -r["confidence"]))[:8]

# ---------- 4. Classification & regression (time-based split = predicting the future) ----------
F = ["area_code", "hour", "dow", "weekend", "rain_mm", "event", "lag_24", "lag_168"]
cut = df.timestamp.max() - pd.Timedelta(days=21)
tr, te = df[df.timestamp <= cut], df[df.timestamp > cut]
clfs = {"Random Forest": RandomForestClassifier(200, n_jobs=-1, random_state=1),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=1),
        "k-NN": KNeighborsClassifier(15)}
sc = StandardScaler().fit(tr[F]); cres, best_c, best_acc = [], None, -1
for name, m in clfs.items():
    Xa, Xb = (sc.transform(tr[F]), sc.transform(te[F])) if name == "k-NN" else (tr[F], te[F])
    m.fit(Xa, tr.level); p = m.predict(Xb)
    acc = accuracy_score(te.level, p)
    cres.append({"model": name, "accuracy": round(acc, 4), "f1": round(f1_score(te.level, p, average="weighted"), 4)})
    if acc > best_acc: best_acc, best_c, best_pred = acc, name, p
cm = confusion_matrix(te.level, best_pred, labels=LEVELS).tolist()
rf = clfs["Random Forest"]; imp = sorted(zip(F, rf.feature_importances_), key=lambda x: -x[1])

regs = {"Gradient Boosting": GradientBoostingRegressor(random_state=1),
        "Random Forest": RandomForestRegressor(200, n_jobs=-1, random_state=1),
        "Linear Regression": LinearRegression()}
rres, preds = [], {}
for name, m in regs.items():
    m.fit(tr[F], tr.congestion_idx); p = m.predict(te[F]); preds[name] = p
    rres.append({"model": name, "r2": round(r2_score(te.congestion_idx, p), 4),
                 "mae": round(mean_absolute_error(te.congestion_idx, p), 2),
                 "rmse": round(mean_squared_error(te.congestion_idx, p) ** .5, 2)})
best_r = max(rres, key=lambda r: r["r2"])["model"]; reg = regs[best_r]

# Optional Shogun check (library is hard to install on some systems)
try:
    import shogun as sg
    Xs = sg.features(sc.transform(tr[F]).T.astype(float)); Ys = sg.MulticlassLabels(pd.Categorical(tr.level, LEVELS).codes.astype(float))
    knn = sg.machine("KNN", k=15, distance=sg.distance("EuclideanDistance"), labels=Ys); knn.train(Xs)
    sp = knn.apply(sg.features(sc.transform(te[F]).T.astype(float))).get("labels")
    cres.append({"model": "k-NN (Shogun)", "accuracy": round(accuracy_score(pd.Categorical(te.level, LEVELS).codes, sp), 4), "f1": None})
except Exception:
    pass

# ---------- 5. Clustering: area-hour behaviour profiles ----------
prof = df.groupby(["area", "hour"]).agg(idx=("congestion_idx", "mean"), spd=("avg_speed", "mean"), vol=("volume", "mean")).reset_index()
Z = StandardScaler().fit_transform(prof[["idx", "spd", "vol"]])
sil = {k: silhouette_score(Z, KMeans(k, n_init=10, random_state=1).fit_predict(Z)) for k in (3, 4, 5)}
k = max(sil, key=sil.get); km = KMeans(k, n_init=10, random_state=1).fit(Z); prof["cluster"] = km.labels_
order = prof.groupby("cluster").idx.mean().sort_values().index.tolist()
names = ["Free-flow", "Moderate", "Heavy", "Gridlock", "Extreme"][:k]
prof["cluster"] = prof.cluster.map({c: names[i] for i, c in enumerate(order)})
clusters = [{"name": nm, "hours": int((prof.cluster == nm).sum()), "idx": round(prof[prof.cluster == nm].idx.mean(), 1),
             "speed": round(prof[prof.cluster == nm].spd.mean(), 1)} for nm in names if (prof.cluster == nm).any()]

# ---------- 6. Dashboard payload ----------
r1 = lambda x: [round(float(v), 1) for v in x]
heat = df.pivot_table(index="area", columns="hour", values="congestion_idx", aggfunc="mean").loc[area_names]
trend = {a: {"weekday": r1(df[(df.area == a) & (df.weekend == 0)].groupby("hour").congestion_idx.mean()),
             "weekend": r1(df[(df.area == a) & (df.weekend == 1)].groupby("hour").congestion_idx.mean())} for a in area_names}
te = te.assign(pred=preds[best_r])
last7 = te[te.timestamp > te.timestamp.max() - pd.Timedelta(days=7)]
series = {a: {"actual": r1(last7[last7.area == a].congestion_idx), "pred": r1(last7[last7.area == a].pred)} for a in area_names}
mean_idx, lagd = df.groupby(["area", "hour", "weekend"]).congestion_idx.mean(), None
grid = {}
for a_i, a in enumerate(area_names):
    for wk in (0, 1):
        for rain in (0, 1):
            X = pd.DataFrame([[a_i, h, 5 if wk else 2, wk, 6.0 * rain, 0, mean_idx[(a, h, wk)], mean_idx[(a, h, wk)]] for h in range(24)], columns=F)
            grid[f"{a_i}|{wk}|{rain}"] = r1(reg.predict(X))
stats = df.groupby("area").agg(avg=("congestion_idx", "mean"), peak=("congestion_idx", "max"), speed=("avg_speed", "mean"),
                               heavy=("level", lambda s: s.isin(["High", "Severe"]).mean() * 100)).loc[area_names].round(1)
veh = {"total": [int(df[v].sum()) for v in VTYPES], "byArea": {a: [int(df[df.area == a][v].sum()) for v in VTYPES] for a in area_names}}
peaks = {a: [int(h) for h in heat.loc[a].sort_values(ascending=False).index[:2]] for a in area_names}
city = df.groupby("hour").congestion_idx.mean()
out = {"areas": area_names, "levels": LEVELS, "vtypes": VTYPES, "rows": int(len(df)), "train": int(len(tr)), "test": int(len(te)),
       "heat": [r1(heat.loc[a]) for a in area_names], "trend": trend, "series": series, "grid": grid,
       "stats": stats.to_dict("list"), "vehicles": veh, "peaks": peaks, "cityPeak": [int(h) for h in city.sort_values(ascending=False).index[:3]],
       "classifiers": cres, "bestClf": best_c, "confusion": cm, "importance": [[a, round(b, 3)] for a, b in imp],
       "regressors": rres, "bestReg": best_r, "clusters": clusters, "silhouette": {str(a): round(b, 3) for a, b in sil.items()},
       "rules": rules, "levelCounts": df.level.value_counts().reindex(LEVELS).fillna(0).astype(int).tolist()}
(ROOT / "results.json").write_text(json.dumps(out))
html = (ROOT / "dashboard_template.html").read_text().replace("__DATA__", json.dumps(out, separators=(",", ":")))
html = html.replace("__CHARTJS__", (ROOT / "vendor/chart.umd.js").read_text())  # Chart.js bundled: works offline
(ROOT / "dashboard.html").write_text(html)
print(f"rows={len(df)} best classifier={best_c} acc={best_acc:.3f} best regressor={best_r}")
for r in cres + rres: print(r)
print("clusters k=", k, "rules:", len(rules))
