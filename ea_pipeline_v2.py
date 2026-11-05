"""
Early-Access game viability pipeline (first pass).

Idea: take games that were tagged Early Access in an OLD Steam snapshot, then look at a
NEWER snapshot to see what happened to them:
    graduated -> no longer tagged Early Access   (success)
    still_ea  -> still tagged Early Access       (stalled)
    removed   -> no longer on the store          (excluded from the main target)
All features come from the OLD snapshot, all labels from the NEW one, so there is no
time leakage.

Usage:
    python ea_pipeline.py --demo                       # sanity-check on synthetic data
    python ea_pipeline.py --old steam.csv --new games.csv --old-date 2019-05-01 --new-date 2025-01-01

!! Column names below are from memory of the Kaggle files. Open both CSVs, check the
!! headers, and fix OLD_COLS / NEW_COLS before running on real data.
"""
import argparse, os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
                                     cross_validate, cross_val_predict)
from scipy.stats import mannwhitneyu
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
import shap

warnings.filterwarnings("ignore")
OUT = "ea_outputs"
os.makedirs(OUT, exist_ok=True)
REPEATS = 3          # CV repeats (set with --repeats)
SUMMARY = {}         # numbers that were only printed before; saved to run_summary.csv

# ---- EDIT THESE to match the real CSV headers --------------------------------------
OLD_COLS = dict(id="appid", name="name", release="release_date", developer="developer",
                genres="genres", tags="steamspy_tags", owners="owners", price="price",
                pos="positive_ratings", neg="negative_ratings", playtime="average_playtime",
                achievements="achievements")
NEW_COLS = dict(id="app_id_real", genres="Genres")   # id is rebuilt by load_new() below
# --------------------------------------------------------------------------------------


def load_new(path):
    """The Fronkon CSV has 1 more data field than header names ('DiscountDLC count' is a merged
    header), so pandas silently uses the real AppID as the row index. Rescue it here."""
    df = pd.read_csv(path, on_bad_lines="skip")
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index().rename(columns={df.index.name or "index": "app_id_real"})
    else:
        df["app_id_real"] = df["AppID"]
    df["app_id_real"] = pd.to_numeric(df["app_id_real"], errors="coerce")
    return df.dropna(subset=["app_id_real"])


# ============================== 1. LABELS ==============================================
def build_labels(old, new, old_date, new_date):
    o, n = OLD_COLS, NEW_COLS
    ea = old[old[o["genres"]].fillna("").str.contains("Early Access", case=False)].copy()
    ea["launch"] = pd.to_datetime(ea[o["release"]], errors="coerce")
    ea = ea.dropna(subset=["launch"])

    new_ea_ids = set(new.loc[new[n["genres"]].fillna("").str.contains("Early Access", case=False), n["id"]])
    new_ids = set(new[n["id"]])
    ea["status"] = np.where(~ea[o["id"]].isin(new_ids), "removed",
                    np.where(ea[o["id"]].isin(new_ea_ids), "still_ea", "graduated"))

    # survival columns. Exit date is unknown for graduated games: it lies somewhere between
    # the two snapshots, so use the midpoint (APPROXIMATION - upgrade with SteamDB later).
    old_date, new_date = pd.Timestamp(old_date), pd.Timestamp(new_date)
    ea["entry_days"] = (old_date - ea["launch"]).dt.days.clip(lower=1)          # delayed entry
    mid = (new_date - old_date).days / 2
    ea["duration_days"] = np.where(ea["status"] == "graduated", ea["entry_days"] + mid,
                                   (new_date - ea["launch"]).dt.days)
    ea["event"] = (ea["status"] == "graduated").astype(int)
    # v2: exit time for graduated AND removed games is imputed at the midpoint between snapshots
    ea["duration_all"] = np.where(ea["status"].isin(["graduated", "removed"]),
                                  ea["entry_days"] + mid, (new_date - ea["launch"]).dt.days)
    ea["target"] = ea["event"]            # 1 = graduated, 0 = still stuck in EA
    return ea


# ============================== 2. FEATURES ============================================
def owners_mid(s):
    def f(x):
        try:
            lo, hi = [float(v) for v in str(x).replace(" ", "").split("-")]
            return (lo + hi) / 2
        except Exception:
            return np.nan
    return s.map(f)


def build_features(ea, old, old_date):
    o = OLD_COLS
    df = ea.copy()
    df["age_days"] = df["entry_days"]
    df["log_price"] = np.log1p(df[o["price"]].fillna(0))
    df["log_owners"] = np.log1p(owners_mid(df[o["owners"]]).fillna(0))
    total = (df[o["pos"]].fillna(0) + df[o["neg"]].fillna(0))
    df["log_reviews"] = np.log1p(total)
    df["pos_ratio"] = (df[o["pos"]].fillna(0) / total.replace(0, np.nan)).fillna(0.5)
    df["reviews_per_month"] = total / (df["age_days"] / 30).clip(lower=1)    # corrects for game age
    df["log_playtime"] = np.log1p(df[o["playtime"]].fillna(0))
    df["achievements"] = df[o["achievements"]].fillna(0)
    df["launch_month"] = df["launch"].dt.month
    df["launch_year"] = df["launch"].dt.year

    # developer track record: other games by the same developer in the snapshot
    dev_counts = old[o["developer"]].value_counts()
    df["dev_other_games"] = df[o["developer"]].map(dev_counts).fillna(1) - 1

    # genre saturation: how many EA games shared the first tag in the same launch year
    # first tag that is NOT "Early Access" (every game in this sample carries that tag, and in the
    # SteamSpy list it is usually first, which made main_tag identical for all games in v1)
    def first_real_tag(x):
        for t in str(x).split(";"):
            if t.strip() and t.strip().lower() != "early access":
                return t.strip()
        return "none"
    df["main_tag"] = df[o["tags"]].map(first_real_tag)
    df["tag_year_saturation"] = df.groupby(["main_tag", "launch_year"])["main_tag"].transform("count")

    # top-10 tags as binary columns
    top = df["main_tag"].value_counts().head(10).index
    for t in top:
        df["tag_" + str(t).replace(" ", "_")] = (df["main_tag"] == t).astype(int)

    feats = ["age_days", "log_price", "log_owners", "log_reviews", "pos_ratio", "reviews_per_month",
             "log_playtime", "achievements", "launch_month", "dev_other_games", "tag_year_saturation"] \
            + [c for c in df.columns if c.startswith("tag_") and c != "tag_year_saturation"]
    return df, feats


# ============================== 3. MODELS ==============================================
def make_models(pos_weight):
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(400, min_samples_leaf=3, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=pos_weight,
                                 eval_metric="logloss", random_state=42),
    }


def cv_eval(X, y, tag):
    """5-fold stratified CV repeated REPEATS times: more trustworthy than one 80/20 split."""
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=42)
    pw = (y == 0).sum() / max((y == 1).sum(), 1)
    rows = []
    for name, m in make_models(pw).items():
        r = cross_validate(m, X, y, cv=skf, scoring={"roc": "roc_auc", "pr": "average_precision", "f1": "f1"})
        rows.append(dict(setting=tag, model=name,
                         ROC_AUC=r["test_roc"].mean(), ROC_AUC_sd=r["test_roc"].std(),
                         PR_AUC=r["test_pr"].mean(), F1=r["test_f1"].mean()))
    return pd.DataFrame(rows).round(3)


def run_classification(df, feats):
    d = df[df["status"].isin(["graduated", "still_ea"])]
    X, y = d[feats].fillna(0), d["target"]
    SUMMARY["n_classification"] = len(d); SUMMARY["graduated_rate_classification(PR-AUC baseline)"] = round(float(y.mean()), 4)
    print(f"\nClassification set: {len(d)} games | graduated rate = {y.mean():.1%} (this is the PR-AUC baseline)")

    # (a) single hold-out split (also used for SHAP)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    rows, fitted = [], {}
    for name, m in make_models(pw).items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        rows.append(dict(model=name, ROC_AUC=roc_auc_score(yte, p), PR_AUC=average_precision_score(yte, p),
                         F1=f1_score(yte, p > 0.5)))
        fitted[name] = m
    res = pd.DataFrame(rows).round(3)
    print("\nHold-out split (80/20):"); print(res.to_string(index=False))
    res.to_csv(f"{OUT}/model_comparison.csv", index=False)

    # (b) 5-fold CV
    cv = cv_eval(X, y, "main")
    print("\n5-fold cross-validation (use THIS table in the report):"); print(cv.to_string(index=False))
    cv.to_csv(f"{OUT}/cv_comparison.csv", index=False)

    # SHAP on XGBoost
    sv = shap.TreeExplainer(fitted["XGBoost"]).shap_values(Xte)
    shap.summary_plot(sv, Xte, show=False)
    plt.tight_layout(); plt.savefig(f"{OUT}/shap_summary.png", dpi=150); plt.close()


def run_sensitivity(df, feats):
    """Same models, but removed/delisted games are counted as FAILURES instead of dropped."""
    d = df.copy()
    X, y = d[feats].fillna(0), (d["status"] == "graduated").astype(int)
    print(f"\nSensitivity: removed games counted as failures | n = {len(d)} | graduated rate = {y.mean():.1%}")
    cv = cv_eval(X, y, "removed=failure")
    print(cv.to_string(index=False))
    cv.to_csv(f"{OUT}/sensitivity_removed_as_failure.csv", index=False)


def run_survival(df, feats):
    cols = ["log_price", "log_owners", "pos_ratio", "reviews_per_month", "dev_other_games", "tag_year_saturation"]
    raw = df[df["status"].isin(["graduated", "still_ea"])][cols + ["duration_days", "event", "entry_days"]].fillna(0).copy()
    d = raw.copy()
    d[cols] = StandardScaler().fit_transform(d[cols])

    # entry_col handles delayed entry: games that left EA BEFORE the old snapshot are not in our data
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(d, duration_col="duration_days", event_col="event", entry_col="entry_days")
    print("\nCox PH concordance:", round(cph.concordance_index_, 3))
    SUMMARY["cox_graduation_concordance"] = round(float(cph.concordance_index_), 3)
    summ = cph.summary[["coef", "exp(coef)", "p"]].round(3)
    print(summ); summ.to_csv(f"{OUT}/cox_summary.csv")
    cph.plot(); plt.tight_layout(); plt.savefig(f"{OUT}/cox_coefficients.png", dpi=150); plt.close()

    # proportional-hazards assumption test. lifelines can't run it on a model with delayed entry,
    # so this is run on a copy WITHOUT entry_col -> treat as an APPROXIMATE diagnostic.
    try:
        cph_diag = CoxPHFitter(penalizer=0.1).fit(d.drop(columns=["entry_days"]),
                                                  duration_col="duration_days", event_col="event")
        ph = proportional_hazard_test(cph_diag, d.drop(columns=["entry_days"]), time_transform="rank")
        print("\nProportional hazards test, approximate (p < 0.05 = effect may change over time):")
        print(ph.summary[["test_statistic", "p"]].round(3))
        ph.summary[["test_statistic", "p"]].round(3).to_csv(f"{OUT}/cox_ph_assumption_test.csv")
    except Exception as e:
        print("\nPH assumption test skipped:", e)

    # Kaplan-Meier: probability a game is STILL in Early Access, high vs low (median split)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (col, label) in zip(axes, [("pos_ratio", "positive review ratio"),
                                       ("reviews_per_month", "reviews per month")]):
        hi = raw[col] >= raw[col].median()
        for mask, nm in [(hi, f"high {label}"), (~hi, f"low {label}")]:
            km = KaplanMeierFitter()
            km.fit(raw.loc[mask, "duration_days"] / 365.25, raw.loc[mask, "event"],
                   entry=raw.loc[mask, "entry_days"] / 365.25, label=nm)
            km.plot_survival_function(ax=ax)
        ax.set_xlabel("years since Early Access launch"); ax.set_title(label)
    axes[0].set_ylabel("P(still in Early Access)")
    plt.tight_layout(); plt.savefig(f"{OUT}/km_curves.png", dpi=150); plt.close()



# ============================== 4. v2 ADDITIONS ========================================
SURV_COLS = ["log_price", "log_owners", "pos_ratio", "reviews_per_month", "dev_other_games", "tag_year_saturation"]


def run_competing_risks(df):
    """Cause-specific Cox: time-to-graduation vs time-to-removal (proposal: 'time-to-full-release versus
    time-to-abandonment'). In each model the OTHER outcome is censored at its midpoint-imputed exit time.
    Removal dates are as approximate as graduation dates -> state this in the report."""
    d = df[SURV_COLS + ["duration_all", "entry_days", "status"]].copy()
    d[SURV_COLS] = StandardScaler().fit_transform(d[SURV_COLS].fillna(0))
    fits = {}
    for name, ev in [("graduation", "graduated"), ("removal", "removed")]:
        m = d.drop(columns="status").copy()
        m["event"] = (d["status"] == ev).astype(int)
        n_ev = int(m["event"].sum())
        SUMMARY[f"cause_specific_events_{name}"] = n_ev
        if n_ev < 30:
            print(f"\nCause-specific Cox ({name}) skipped: only {n_ev} events"); continue
        cph = CoxPHFitter(penalizer=0.1).fit(m, duration_col="duration_all", event_col="event", entry_col="entry_days")
        SUMMARY[f"cause_specific_concordance_{name}"] = round(float(cph.concordance_index_), 3)
        s = cph.summary[["coef", "exp(coef)", "p"]].round(3)
        print(f"\nCause-specific Cox: {name} (events = {n_ev}, concordance = {cph.concordance_index_:.3f})")
        print(s); s.to_csv(f"{OUT}/cox_cause_specific_{name}.csv")
        fits[name] = cph
    if fits:
        fig, ax = plt.subplots(figsize=(7.5, 4.2)); y = np.arange(len(SURV_COLS))
        for k, (name, cph) in enumerate(fits.items()):
            s = cph.summary.loc[SURV_COLS]
            ax.errorbar(s["exp(coef)"], y + (k - 0.5) * 0.25,
                        xerr=[s["exp(coef)"] - s["exp(coef) lower 95%"], s["exp(coef) upper 95%"] - s["exp(coef)"]],
                        fmt="o", capsize=3, label=name)
        ax.axvline(1, ls="--", c="grey"); ax.set_yticks(y); ax.set_yticklabels(SURV_COLS)
        ax.set_xlabel("hazard ratio per +1 SD (95% CI)"); ax.set_title("Cause-specific hazards: graduation vs removal")
        ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/cox_cause_specific.png", dpi=150); plt.close()


def run_risk_tiers(df, feats):
    """'Financial risk curve': out-of-fold risk score (1 - P(graduate)) -> Low/Medium/High tercile tiers,
    then the REALISED outcome rates and Kaplan-Meier curves per tier. Out-of-fold = no train/test leakage.
    Tiers are rank-based terciles (the score is not a calibrated probability: class_weight='balanced')."""
    X, y = df[feats].fillna(0), (df["status"] == "graduated").astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    p = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
    d = df.assign(risk=1 - p)
    d["risk_tier"] = pd.qcut(d["risk"].rank(method="first"), 3, labels=["Low", "Medium", "High"])
    tab = (pd.crosstab(d["risk_tier"], d["status"], normalize="index") * 100).round(1)
    tab.insert(0, "n", d["risk_tier"].value_counts().sort_index())
    print("\nRealised outcome (%) by predicted risk tier:"); print(tab)
    tab.to_csv(f"{OUT}/risk_tiers.csv")
    cols = [c for c in ["graduated", "still_ea", "removed"] if c in tab.columns]
    tab[cols].plot(kind="bar", stacked=True, figsize=(6, 4)); plt.ylabel("% of games"); plt.xlabel("predicted risk tier")
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(f"{OUT}/risk_tiers.png", dpi=150); plt.close()

    s = d[d["status"].isin(["graduated", "still_ea"])]
    fig, ax = plt.subplots(figsize=(6, 4))
    for tier in ["Low", "Medium", "High"]:
        g = s[s["risk_tier"] == tier]
        km = KaplanMeierFitter()
        km.fit(g["duration_days"] / 365.25, g["event"], entry=g["entry_days"] / 365.25, label=f"{tier} risk (n={len(g)})")
        km.plot_survival_function(ax=ax, ci_show=False)
    ax.set_xlabel("years since Early Access launch"); ax.set_ylabel("P(still in Early Access)")
    ax.set_title("Risk curves by predicted tier (exit times approximate)")
    plt.tight_layout(); plt.savefig(f"{OUT}/risk_curves_by_tier.png", dpi=150); plt.close()


def _col(df, name):
    hits = [c for c in df.columns if str(c).strip().lower() == name.lower()]
    return hits[0] if hits else None


def run_validation(df, new):
    """Do 'graduated' games really look healthier afterwards? Compares OLD -> NEW snapshot change in review
    score, review volume and playtime. DESCRIPTIVE label validation only - these are post-outcome, so they
    must NOT be used as model features (leakage). Needs Positive/Negative/Average playtime forever in NEW."""
    pc, nc, tc = _col(new, "Positive"), _col(new, "Negative"), _col(new, "Average playtime forever")
    if not (pc and nc and tc):
        print("\nValidation skipped. NEW columns containing 'pos/neg/playtime':",
              [c for c in new.columns if any(k in str(c).lower() for k in ("pos", "neg", "playtime"))]); return
    nv = new[[NEW_COLS["id"], pc, nc, tc]].copy()
    for c in (pc, nc, tc): nv[c] = pd.to_numeric(nv[c], errors="coerce")
    tot = nv[pc] + nv[nc]
    nv["pos_ratio_new"] = nv[pc] / tot.replace(0, np.nan)
    nv["log_reviews_new"] = np.log1p(tot)
    nv["log_playtime_new"] = np.log1p(nv[tc])
    m = df[[OLD_COLS["id"], "status", "pos_ratio", "log_reviews", "log_playtime"]].merge(
        nv.drop_duplicates(NEW_COLS["id"]), left_on=OLD_COLS["id"], right_on=NEW_COLS["id"], how="inner")
    m["review_score_change"] = m["pos_ratio_new"] - m["pos_ratio"]
    m["review_volume_growth(log)"] = m["log_reviews_new"] - m["log_reviews"]
    m["playtime_change(log)"] = m["log_playtime_new"] - m["log_playtime"]
    metrics = ["review_score_change", "review_volume_growth(log)", "playtime_change(log)"]
    tab = m.groupby("status")[metrics].median().round(3)
    tab["n"] = m["status"].value_counts()
    g, s = m[m["status"] == "graduated"], m[m["status"] == "still_ea"]
    if len(g) > 10 and len(s) > 10:
        tab.loc["MWU p (grad vs still_ea)", metrics] = [round(mannwhitneyu(g[c].dropna(), s[c].dropna()).pvalue, 4) for c in metrics]
    print("\nOLD->NEW change by outcome (median) - label validation, NOT features:"); print(tab)
    tab.to_csv(f"{OUT}/label_validation.csv")

# ============================== DEMO DATA ==============================================
def make_demo():
    rng = np.random.default_rng(0)
    n = 4000
    launch = pd.to_datetime("2013-01-01") + pd.to_timedelta(rng.integers(0, 2300, n), unit="D")
    tags = rng.choice(["Survival", "Action", "RPG", "Strategy", "Simulation", "Indie", "Horror"], n)
    old = pd.DataFrame(dict(
        appid=np.arange(n), name=[f"g{i}" for i in range(n)], release_date=launch.strftime("%Y-%m-%d"),
        developer=rng.choice([f"dev{i}" for i in range(1500)], n),
        genres=np.where(rng.random(n) < 0.5, "Action;Indie;Early Access", "Action;Indie"),
        steamspy_tags=tags, owners=rng.choice(["0-20000", "20000-50000", "100000-200000", "500000-1000000"], n),
        price=rng.choice([4.99, 9.99, 14.99, 19.99, 29.99], n), positive_ratings=rng.integers(0, 5000, n),
        negative_ratings=rng.integers(0, 1500, n), average_playtime=rng.integers(0, 3000, n),
        achievements=rng.integers(0, 100, n)))
    ea = old["genres"].str.contains("Early Access")
    z = 0.0015 * old["positive_ratings"] - 0.002 * old["negative_ratings"] + 0.5 * np.log1p(old["average_playtime"]) - 4.0
    grad = rng.random(n) < 1 / (1 + np.exp(-z))
    removed = rng.random(n) < 0.08
    new_rows = old[~removed].copy()
    new_rows["genres"] = np.where(ea[~removed] & ~grad[~removed], "Action;Indie;Early Access", "Action;Indie")
    boost = np.where(grad[~removed], 1.4, 1.0)
    new_rows["Positive"] = (new_rows["positive_ratings"] * boost).astype(int)
    new_rows["Negative"] = new_rows["negative_ratings"]
    new_rows["Average playtime forever"] = (new_rows["average_playtime"] * boost).astype(int)
    new = new_rows.rename(columns={"appid": "AppID", "genres": "Genres"})[["AppID", "Genres", "Positive", "Negative", "Average playtime forever"]]
    return old, new


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--old"); ap.add_argument("--new")
    ap.add_argument("--old-date", default="2019-05-01")
    ap.add_argument("--new-date", default="2025-01-01")
    ap.add_argument("--repeats", type=int, default=3, help="CV repeats (use 1 for a quick run)")
    a = ap.parse_args()
    REPEATS = a.repeats

    if a.demo:
        old, new = make_demo()
        new = new.rename(columns={"AppID": "app_id_real"})
    else:
        old, new = pd.read_csv(a.old), load_new(a.new)

    ea = build_labels(old, new, a.old_date, a.new_date)
    print("Status counts:\n", ea["status"].value_counts())
    SUMMARY.update({f"n_{k}": int(v) for k, v in ea["status"].value_counts().items()})
    df, feats = build_features(ea, old, a.old_date)
    df.to_csv(f"{OUT}/ea_labeled_dataset.csv", index=False)       # small file you can push to GitHub
    run_classification(df, feats)
    run_sensitivity(df, feats)
    run_survival(df, feats)
    run_competing_risks(df)      # v2: release vs abandonment
    run_risk_tiers(df, feats)    # v2: financial risk curve
    run_validation(df, new)      # v2: label validation (engagement / review change)
    pd.Series(SUMMARY).to_csv(f"{OUT}/run_summary.csv", header=["value"])
    print(f"\nOutputs saved in ./{OUT}/")
