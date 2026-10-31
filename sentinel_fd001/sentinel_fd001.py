"""SENTINEL: FD001 baseline.
Data -> features -> anomaly + RUL -> rule-based triage -> work-order drafts -> MLflow + retraining demo.
Run:  python sentinel_fd001.py --data DATA_DIR --out results
DATA_DIR must hold train_FD001.txt, test_FD001.txt, RUL_FD001.txt (NASA C-MAPSS).
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, IsolationForest
import mlflow

SEED = 42
RUL_CAP = 125          # piecewise-linear RUL target (standard for C-MAPSS)
WIN = 10               # rolling window (cycles)
COLS = ["unit", "cycle"] + [f"op{i}" for i in (1, 2, 3)] + [f"s{i}" for i in range(1, 22)]
# triage thresholds (cycles of remaining life); tuned/validated, see README
T_CRIT, T_HIGH, T_WATCH = 30, 60, 100
SUSTAINED = 3          # consecutive anomaly flags that escalate severity
DISAGREE = 25          # model disagreement (cycles) that sends a case to a human


def load(data_dir):
    tr = pd.read_csv(f"{data_dir}/train_FD001.txt", sep=r"\s+", header=None, names=COLS)
    te = pd.read_csv(f"{data_dir}/test_FD001.txt", sep=r"\s+", header=None, names=COLS)
    rul = pd.read_csv(f"{data_dir}/RUL_FD001.txt", header=None, names=["rul"])["rul"].values
    return tr, te, rul


def add_rul(df):
    df = df.copy()
    df["rul_true"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    df["rul"] = df["rul_true"].clip(upper=RUL_CAP)
    return df


def choose_sensors(train):
    """Data cleaning: drop operating settings and flat (near-constant) sensors."""
    s = [c for c in train.columns if c.startswith("s")]
    keep = [c for c in s if train[c].std() > 1e-3 and train[c].nunique() > 5]
    return keep


class Prep:
    """Impute -> standardise (fit on train only) -> rolling features."""
    def __init__(self, sensors):
        self.sensors = sensors

    def fit(self, df):
        self.med = df[self.sensors].median()
        self.mu = df[self.sensors].mean()
        self.sd = df[self.sensors].std().replace(0, 1)
        return self

    def transform(self, df):
        d = df.copy()
        d[self.sensors] = d[self.sensors].fillna(self.med)           # missing values
        d[self.sensors] = (d[self.sensors] - self.mu) / self.sd      # scale
        g = d.groupby("unit")[self.sensors]
        mean = g.rolling(WIN, min_periods=1).mean().reset_index(level=0, drop=True)    # denoise
        std = g.rolling(WIN, min_periods=2).std().reset_index(level=0, drop=True).fillna(0)
        slope = (mean - mean.groupby(d["unit"]).shift(WIN - 1)) / (WIN - 1)             # trend
        slope = slope.fillna(0)
        feats = pd.concat([mean.add_suffix("_mean"), std.add_suffix("_std"), slope.add_suffix("_slope")], axis=1)
        feats = feats.loc[d.index]            # keep row order identical to the input frame
        feats["cycle"] = d["cycle"].values
        return d, feats


def nasa_score(pred, true):
    d = np.asarray(pred) - np.asarray(true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def psi(a, b, bins=10):
    """Population Stability Index between reference a and new b."""
    edges = np.quantile(a, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a) + 1e-6
    pb = np.histogram(b, edges)[0] / len(b) + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def severity(rul_pred, flags_run):
    if rul_pred <= T_CRIT: return "CRITICAL"
    if rul_pred <= T_HIGH or flags_run >= SUSTAINED: return "HIGH"
    if rul_pred <= T_WATCH: return "WATCH"
    return "HEALTHY"


def sev_from_true(r):
    return "CRITICAL" if r <= T_CRIT else "HIGH" if r <= T_HIGH else "WATCH" if r <= T_WATCH else "HEALTHY"


def main(a):
    os.makedirs(a.out, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(a.out)}/mlflow.db")
    mlflow.set_experiment("sentinel_fd001")
    train, test, rul_true_test = load(a.data)
    train = add_rul(train)
    sensors = choose_sensors(train)
    print("sensors kept:", len(sensors), "dropped:", sorted(set(c for c in train.columns if c.startswith('s')) - set(sensors)))

    # engine-level split of training data for validation (threshold tuning, drift demo)
    units = np.array(sorted(train.unit.unique()))
    rng = np.random.default_rng(SEED); rng.shuffle(units)
    val_units, fit_units = units[:20], units[20:]
    fit_df, val_df = train[train.unit.isin(fit_units)], train[train.unit.isin(val_units)]

    prep = Prep(sensors).fit(fit_df)
    Xn_fit, F_fit = prep.transform(fit_df)
    Xn_val, F_val = prep.transform(val_df)

    # ---------- RUL models ----------
    def fit_models(F, y):
        gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                           l2_regularization=1.0, random_state=SEED).fit(F, y)
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.3,
                                   n_jobs=-1, random_state=SEED).fit(F, y)
        return gb, rf

    with mlflow.start_run(run_name="v1_baseline") as run:
        gb, rf = fit_models(F_fit, fit_df["rul"])
        pv = 0.5 * (gb.predict(F_val) + rf.predict(F_val))
        # last-cycle-style validation: score every validation row and the cutoff points
        val_rmse = rmse(pv, val_df["rul"])
        mlflow.log_params({"win": WIN, "rul_cap": RUL_CAP, "n_sensors": len(sensors), "models": "HistGB+RF avg"})
        mlflow.log_metric("val_rmse_all_cycles", val_rmse)

        # ---------- anomaly detection ----------
        # Healthy baseline = first 40 cycles of each fit engine (early life)
        base = Xn_fit[fit_df["cycle"].values <= 40][sensors]
        iso = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED).fit(base)
        raw_val = -iso.score_samples(Xn_val[sensors])
        raw_base = -iso.score_samples(base)
        thr = float(np.quantile(raw_base, 0.99))       # 1% false-alarm budget on healthy early life

        def flags_for(Xn, F):
            sc = -iso.score_samples(Xn[sensors])
            # smooth with rolling mean so single noisy points do not fire
            s = pd.Series(sc, index=Xn.index).groupby(Xn["unit"]).transform(lambda x: x.rolling(5, min_periods=1).mean())
            return s.values, (s.values > thr).astype(int)

        sc_val, fl_val = flags_for(Xn_val, F_val)
        v = val_df.assign(score=sc_val, flag=fl_val)
        # lead time: RUL (cycles to failure) at the first flag that stays on for SUSTAINED cycles
        lead, missed = [], 0
        for u, g in v.groupby("unit"):
            run = g["flag"].rolling(SUSTAINED).sum().values
            idx = np.where(run >= SUSTAINED)[0]
            if len(idx) == 0: missed += 1; continue
            lead.append(int(g["rul_true"].values[idx[0]]))
        healthy_early = v[v["rul_true"] > 150]
        far = float(healthy_early["flag"].mean()) if len(healthy_early) else float("nan")
        mlflow.log_metrics({"anomaly_median_lead_cycles": float(np.median(lead)), "anomaly_missed_engines": missed,
                            "anomaly_false_flag_rate_healthy": far})

        # ---------- official test set: last cycle of each engine ----------
        Xn_te, F_te = prep.transform(test)
        last = test.groupby("unit")["cycle"].transform("max") == test["cycle"]
        Fl = F_te[last.values]; ul = test.loc[last, "unit"].values
        p_gb, p_rf = gb.predict(Fl), rf.predict(Fl)
        p = np.clip(0.5 * (p_gb + p_rf), 0, RUL_CAP)
        test_rmse, test_score = rmse(p, rul_true_test), nasa_score(p, rul_true_test)
        rmse_gb, rmse_rf = rmse(p_gb, rul_true_test), rmse(p_rf, rul_true_test)
        mlflow.log_metrics({"test_rmse": test_rmse, "test_nasa_score": test_score,
                            "test_rmse_histgb": rmse_gb, "test_rmse_rf": rmse_rf})

        # ---------- triage + work orders on test engines ----------
        sc_te, fl_te = flags_for(Xn_te, F_te)
        t = test.assign(flag=fl_te, score=sc_te)
        recs = []
        for i, u in enumerate(ul):
            g = t[t.unit == u]
            # consecutive flags ending at last cycle
            f = g["flag"].values; k = 0
            for x in f[::-1]:
                if x == 1: k += 1
                else: break
            sev = severity(p[i], k)
            gap = abs(p_gb[i] - p_rf[i])
            conflict = (sev == "HEALTHY" and k >= SUSTAINED)
            review = bool(gap > DISAGREE or conflict)
            # evidence: sensors with largest standardised deviation from healthy baseline at last cycle
            row = Xn_te[(t.unit == u).values].iloc[-1][sensors]
            top = row.abs().sort_values(ascending=False).head(3)
            recs.append({"unit": int(u), "cycles_observed": int(g["cycle"].max()), "pred_rul": round(float(p[i]), 1),
                         "true_rul": int(rul_true_test[i]), "anomaly_flags_in_a_row": k, "severity": sev,
                         "needs_human_review": review,
                         "evidence": "; ".join(f"{s} ({row[s]:+.2f} sd)" for s in top.index),
                         "recommended_action": {"CRITICAL": "Inspect now; take out of service if confirmed",
                                                "HIGH": "Schedule inspection in next maintenance window",
                                                "WATCH": "Monitor; re-check after next 20 cycles",
                                                "HEALTHY": "No action"}[sev],
                         "status": "DRAFT - awaiting approval" if sev != "HEALTHY" or review else "no order"})
        wo = pd.DataFrame(recs)
        wo["true_severity"] = wo["true_rul"].map(sev_from_true)
        wo.to_csv(f"{a.out}/work_orders_test_fleet.csv", index=False)
        order = ["HEALTHY", "WATCH", "HIGH", "CRITICAL"]
        conf = pd.crosstab(wo["true_severity"], wo["severity"]).reindex(index=order, columns=order, fill_value=0)
        conf.to_csv(f"{a.out}/triage_confusion.csv")
        crit_true = wo[wo.true_severity == "CRITICAL"]
        crit_recall = float((crit_true["severity"].isin(["CRITICAL", "HIGH"])).mean()) if len(crit_true) else float("nan")
        missed_crit = int((crit_true["severity"].isin(["HEALTHY", "WATCH"])).sum())
        sev_acc = float((wo.severity == wo.true_severity).mean())
        within_one = float((abs(wo.severity.map(order.index) - wo.true_severity.map(order.index)) <= 1).mean())
        mlflow.log_metrics({"triage_exact_acc": sev_acc, "triage_within_one_level": within_one,
                            "critical_caught_as_critical_or_high": crit_recall})
        mlflow.log_artifact(f"{a.out}/work_orders_test_fleet.csv")

        # ---------- retraining demo: new data batch arrives ----------
        # v1: trained on 60 engines. New batch of 20 engines arrives -> PSI drift check -> retrain -> v2.
        base_units, new_units = fit_units[:60], fit_units[60:80]
    res_v1 = None
    with mlflow.start_run(run_name="retrain_demo"):
        old = fit_df[fit_df.unit.isin(base_units)]; new = fit_df[fit_df.unit.isin(new_units)]
        p1 = Prep(sensors).fit(old); _, F_old = p1.transform(old); _, F_new = p1.transform(new)
        _, F_val1 = p1.transform(val_df)
        gb1, rf1 = fit_models(F_old, old["rul"])
        v1 = rmse(0.5 * (gb1.predict(F_val1) + rf1.predict(F_val1)), val_df["rul"])
        drifts = {s: psi(F_old[f"{s}_mean"].values, F_new[f"{s}_mean"].values) for s in sensors}
        max_psi = max(drifts.values()); trig = max_psi > 0.2
        both = pd.concat([old, new]); p2 = Prep(sensors).fit(both); _, F_both = p2.transform(both); _, F_val2 = p2.transform(val_df)
        gb2, rf2 = fit_models(F_both, both["rul"])
        v2 = rmse(0.5 * (gb2.predict(F_val2) + rf2.predict(F_val2)), val_df["rul"])
        mlflow.log_params({"engines_v1": len(base_units), "engines_new_batch": len(new_units)})
        mlflow.log_metrics({"max_feature_psi": max_psi, "val_rmse_v1": v1, "val_rmse_v2_after_retrain": v2})
        res_v1 = {"val_rmse_v1_60_engines": v1, "val_rmse_v2_80_engines": v2, "max_psi": max_psi,
                  "psi_above_0.2_triggers_retrain": bool(trig)}

    # ---------- plots ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(rul_true_test, p, s=18, c="#1799D6", edgecolor="#0B2A5B", lw=0.4)
    ax[0].plot([0, 140], [0, 140], "--", c="#D64545"); ax[0].set_xlabel("True RUL (cycles)"); ax[0].set_ylabel("Predicted RUL (cycles)")
    ax[0].set_title(f"FD001 test engines: RMSE {test_rmse:.1f}, NASA score {test_score:,.0f}")
    u0 = int(v.groupby("unit")["rul_true"].max().idxmax())
    g = v[v.unit == u0]
    ax[1].plot(g["cycle"], g["score"], c="#0B2A5B", lw=1.3, label="anomaly score")
    ax[1].axhline(thr, c="#D64545", ls="--", label="alert threshold")
    ax[1].set_xlabel("Cycle"); ax[1].set_ylabel("Isolation Forest score"); ax[1].legend()
    ax[1].set_title(f"Validation engine {u0}: score rises before failure")
    plt.tight_layout(); plt.savefig(f"{a.out}/fd001_results.png", dpi=160); plt.close()

    out = {"sensors_kept": sensors, "val_rmse_all_cycles": val_rmse,
           "test": {"rmse": test_rmse, "nasa_score": test_score, "rmse_histgb": rmse_gb, "rmse_rf": rmse_rf},
           "anomaly_validation": {"median_lead_cycles_before_failure": float(np.median(lead)),
                                  "min_lead": int(min(lead)), "max_lead": int(max(lead)),
                                  "engines_never_flagged": missed, "false_flag_rate_when_RUL_gt_150": far},
           "triage_test_fleet": {"exact_accuracy": sev_acc, "within_one_level": within_one,
                                 "true_critical_engines": int(len(crit_true)),
                                 "true_critical_caught_as_critical_or_high": crit_recall,
                                 "true_critical_missed_as_healthy_or_watch": missed_crit,
                                 "needs_human_review": int(wo.needs_human_review.sum()),
                                 "orders_drafted": int((wo.status != "no order").sum())},
           "retraining_demo": res_v1}
    json.dump(out, open(f"{a.out}/metrics.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="results")
    main(ap.parse_args())
