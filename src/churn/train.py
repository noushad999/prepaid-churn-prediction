"""Training with out-of-time validation, baselines, business evaluation and artifact export."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import evaluate as ev
from churn.campaign import campaign_summary, expected_net, profit_curve
from churn.explain import FAMILY_LABELS, global_family_importance
from churn.features import CATEGORICAL, LABEL, build_many, build_snapshot, feature_columns
from churn.model import ChurnModel
from churn.monitor import build_reference, save_reference
from churn.simulate import load_tables


class Timer:
    def __init__(self):
        self.marks: dict[str, float] = {}

    def __call__(self, name: str):
        timer = self

        class _T:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                timer.marks[name] = round(time.perf_counter() - self.t0, 2)
                print(f"  [{timer.marks[name]:>6.1f}s] {name}")

        return _T()


def _to_pandas(df: pl.DataFrame, features: list[str], categories: dict[str, list[str]]) -> pd.DataFrame:
    X = df.select(features).to_pandas()
    for col, levels in categories.items():
        X[col] = pd.Categorical(X[col], categories=levels)
    return X


def _logreg_baseline(X_tr: pd.DataFrame, y_tr: np.ndarray, seed: int, max_rows: int = 300_000):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_tr), min(max_rows, len(X_tr)), replace=False)
    num = [c for c in X_tr.columns if c not in CATEGORICAL]
    pre = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    model = make_pipeline(pre, LogisticRegression(max_iter=500, C=0.5))
    model.fit(X_tr.iloc[idx], y_tr[idx])
    return model


def run(cfg: dict) -> dict:
    paths, tcfg, ccfg = cfg["paths"], cfg["training"], cfg["churn"]
    reports = Path(paths["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    Path(paths["features_dir"]).mkdir(parents=True, exist_ok=True)
    timer = Timer()
    snap_kw = dict(lookback_months=ccfg["lookback_months"], inactivity_months=ccfg["inactivity_months"])

    print("Training churn model")
    with timer("load raw tables"):
        tables = load_tables(paths["raw_dir"])
    with timer("build feature snapshots"):
        train = build_many(tables, tcfg["train_cutoffs"], **snap_kw)
        valid = build_snapshot(tables, tcfg["valid_cutoff"], **snap_kw)
        test = build_snapshot(tables, tcfg["test_cutoff"], **snap_kw)
        for name, df in (("train", train), ("valid", valid), ("test", test)):
            df.write_parquet(Path(paths["features_dir"]) / f"{name}.parquet")

    features = feature_columns(train)
    categories = {c: sorted(train[c].drop_nulls().unique().to_list()) for c in CATEGORICAL}
    X_tr, X_va, X_te = (_to_pandas(d, features, categories) for d in (train, valid, test))
    y_tr, y_va, y_te = (d[LABEL].to_numpy() for d in (train, valid, test))
    print(
        f"  train {len(y_tr):,} rows ({len(tcfg['train_cutoffs'])} snapshots, churn {y_tr.mean():.2%}) | "
        f"valid {len(y_va):,} ({y_va.mean():.2%}) | test {len(y_te):,} ({y_te.mean():.2%}) | {len(features)} features"
    )

    with timer("train LightGBM"):
        booster = lgb.train(
            tcfg["lgbm"] | {"seed": cfg["seed"]},
            lgb.Dataset(X_tr, y_tr, categorical_feature=CATEGORICAL, free_raw_data=False),
            num_boost_round=tcfg["num_boost_round"],
            valid_sets=[lgb.Dataset(X_va, y_va, categorical_feature=CATEGORICAL)],
            callbacks=[lgb.early_stopping(tcfg["early_stopping_rounds"], verbose=False)],
        )
    with timer("train logistic-regression baseline"):
        logreg = _logreg_baseline(X_tr, y_tr, cfg["seed"])

    with timer("evaluate on out-of-time test month"):
        p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
        p_te = booster.predict(X_te, num_iteration=booster.best_iteration)
        p_lr = logreg.predict_proba(X_te)[:, 1]
        # the pre-ML business rule: longest time since last recharge = highest risk
        p_rule = test["days_since_last_recharge"].fill_null(999).to_numpy().astype(float)
        scores = {"LightGBM": p_te, "Logistic regression": p_lr, "Rule: days since recharge": p_rule}
        metrics = {name: ev.classification_metrics(y_te, s) for name, s in scores.items()}
        gains = ev.gains_table(y_te, p_te)
        gains.to_csv(reports / "gains_test.csv", index=False)

    b = cfg["risk_bands"]
    risk_thresholds = {
        "critical": float(np.quantile(p_va, 1 - b["critical"])),
        "high": float(np.quantile(p_va, 1 - b["critical"] - b["high"])),
        "medium": float(np.quantile(p_va, 1 - b["critical"] - b["high"] - b["medium"])),
    }

    with timer("SHAP reason codes + campaign economics"):
        model = ChurnModel(
            booster,
            {
                "features": features,
                "categories": categories,
                "risk_thresholds": risk_thresholds,
            },
        )
        sample = np.random.default_rng(cfg["seed"]).choice(len(X_te), min(20_000, len(X_te)), replace=False)
        family_importance = global_family_importance(model.contributions(X_te.iloc[sample]), features)
        arpu = test["revenue_avg3m"].to_numpy()
        camp_cfg = cfg["campaign"]
        campaign = campaign_summary(y_te, p_te, arpu, camp_cfg)
        rng = np.random.default_rng(cfg["seed"])
        curves = {
            "LightGBM expected value": profit_curve(y_te, expected_net(p_te, arpu, camp_cfg), arpu, camp_cfg),
            "Rule: days since recharge": profit_curve(y_te, p_rule, arpu, camp_cfg),
            "Random targeting": profit_curve(y_te, rng.random(len(y_te)), arpu, camp_cfg),
        }

    with timer("charts"):
        ev.plot_roc_pr(y_te, scores, reports / "roc_pr.png")
        ev.plot_gains(y_te, scores, reports / "cumulative_gains.png")
        ev.plot_calibration(y_te, p_te, reports / "calibration.png")
        ev.plot_driver_importance(family_importance, FAMILY_LABELS, reports / "churn_drivers.png")
        ev.plot_profit(
            curves,
            campaign["policy_targets"],
            len(y_te),
            camp_cfg["currency"],
            reports / "campaign_profit.png",
        )

    with timer("export artifacts"):
        top_features = sorted(
            zip(features, booster.feature_importance("gain").tolist(), strict=True), key=lambda kv: -kv[1]
        )[:25]
        model.metadata.update(
            {
                "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "label_definition": f"no usage and no recharge in the {ccfg['inactivity_months']} months after the snapshot month",
                "lookback_months": ccfg["lookback_months"],
                "train_cutoffs": tcfg["train_cutoffs"],
                "valid_cutoff": tcfg["valid_cutoff"],
                "test_cutoff": tcfg["test_cutoff"],
                "best_iteration": booster.best_iteration,
                "train_rows": int(len(y_tr)),
                "test_metrics": metrics["LightGBM"],
            }
        )
        model.save(paths["model_dir"])
        save_reference(build_reference(X_va, p_va), paths["model_dir"])

    results = {
        "model_version": model.metadata["model_version"],
        "data": {
            "train_rows": int(len(y_tr)),
            "valid_rows": int(len(y_va)),
            "test_rows": int(len(y_te)),
            "features": len(features),
            "train_churn_rate": float(y_tr.mean()),
            "test_churn_rate": float(y_te.mean()),
        },
        "best_iteration": booster.best_iteration,
        "metrics_test": metrics,
        "risk_thresholds": risk_thresholds,
        "driver_importance": family_importance,
        "top_features_gain": dict(top_features),
        "campaign_test": campaign,
        "timings_s": timer.marks,
    }
    (reports / "metrics.json").write_text(json.dumps(results, indent=2))
    _print_summary(results, gains, camp_cfg["currency"])
    return results


def _print_summary(results: dict, gains: pd.DataFrame, currency: str) -> None:
    print("\nOut-of-time test results")
    print(f"  {'model':<28}{'ROC-AUC':>9}{'PR-AUC':>9}{'KS':>7}{'lift@10%':>10}{'capture@10%':>13}")
    for name, m in results["metrics_test"].items():
        print(
            f"  {name:<28}{m['roc_auc']:>9.3f}{m['pr_auc']:>9.3f}{m['ks']:>7.3f}"
            f"{m['lift_top10']:>10.2f}{m['capture_top10']:>13.1%}"
        )
    print("\nGains table (LightGBM, test month)")
    print(gains.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    c = results["campaign_test"]
    print(
        f"\nCampaign: target {c['policy_targets']:,} ({c['policy_target_share']:.1%}) -> net "
        f"{c['policy_profit']:,.0f} {currency} vs random same size {c['random_targeting_profit_same_k']:,.0f} {currency}"
    )
