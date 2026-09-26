from pathlib import Path

import numpy as np
import polars as pl
from fastapi.testclient import TestClient

from churn.api import create_app
from churn.model import ChurnModel


def test_training_beats_baselines(trained):
    m = trained["metrics_test"]
    assert m["LightGBM"]["roc_auc"] > 0.80
    assert m["LightGBM"]["roc_auc"] > m["Rule: days since recharge"]["roc_auc"]
    assert m["LightGBM"]["lift_top10"] > 3
    assert (
        trained["campaign_test"]["policy_profit"] > trained["campaign_test"]["random_targeting_profit_same_k"]
    )


def test_artifacts_written(small_cfg, trained):
    model_dir = Path(small_cfg["paths"]["model_dir"])  # deployment export of the champion
    version_dir = Path(small_cfg["paths"]["registry_dir"]) / trained["model_version"]
    for f in ("model.txt", "metadata.json", "drift_reference.json", "uplift_metadata.json"):
        assert (model_dir / f).exists() and (version_dir / f).exists()
    for f in ("metrics.json", "roc_pr.png", "cumulative_gains.png", "campaign_profit.png", "gains_test.csv"):
        assert (Path(small_cfg["paths"]["reports_dir"]) / f).exists()


def test_batch_scores(scores):
    assert scores.height > 1000
    assert scores["churn_probability"].is_between(0, 1).all()
    assert set(scores["risk_band"].unique()) <= {"critical", "high", "medium", "low"}
    targets = scores.filter(pl.col("target"))
    assert targets.height > 0
    assert targets["recommended_action"].null_count() == 0  # every targeted sub gets an action
    assert targets["primary_driver"].null_count() == 0
    assert not (targets["sleeping_dog"]).any()  # guard: never contact predicted sleeping dogs
    holdout = targets["arm"] == "holdout"
    assert 0.03 < holdout.mean() < 0.2  # ~10% of every campaign is held out to measure ROI


def test_uplift_trained_and_reported(trained, small_cfg):
    u = trained["uplift_test"]
    assert u is not None and "Risk + sleeping-dog guard" in u["rankings"]
    assert (Path(small_cfg["paths"]["model_dir"]) / "uplift_metadata.json").exists()
    assert (Path(small_cfg["paths"]["reports_dir"]) / "uplift.png").exists()


def test_model_handles_missing_and_unknown_inputs(small_cfg, trained):
    model = ChurnModel.load(small_cfg["paths"]["model_dir"])
    records = [{"division": "Atlantis", "tenure_months": 1, "age": "n/a"}, {}]
    p = model.predict(model.matrix(records))
    assert p.shape == (2,) and ((p > 0) & (p < 1)).all()
    assert np.allclose(p, model.predict(model.frame(records)))


def test_fast_matrix_matches_pandas_path(small_cfg, trained):
    model = ChurnModel.load(small_cfg["paths"]["model_dir"])
    test = pl.read_parquet(Path(small_cfg["paths"]["features_dir"]) / "test.parquet").head(2000)
    fast = model.predict(model.matrix(test))
    via_pandas = model.predict(model.frame(test))
    via_records = model.predict(model.matrix(test.fill_nan(None).to_dicts()))
    assert np.allclose(fast, via_pandas) and np.allclose(fast, via_records)


def test_api(small_cfg, scores):
    scores_path = Path(small_cfg["paths"]["scores_dir"]) / "scores_2025-12.parquet"
    client = TestClient(create_app(small_cfg["paths"]["model_dir"], scores_path))
    assert client.get("/health").json()["batch_scores_loaded"] is True
    assert client.get("/v1/model").json()["n_features"] > 60

    test = pl.read_parquet(Path(small_cfg["paths"]["features_dir"]) / "test.parquet")
    features = test.drop("sub_id", "snapshot", "churn").head(3).fill_nan(None).to_dicts()
    body = {"subscribers": [{"sub_id": i, "features": f} for i, f in enumerate(features)]}
    res = client.post("/v1/predict", json=body)
    assert res.status_code == 200
    preds = res.json()["predictions"]
    assert len(preds) == 3 and all(0 <= p["churn_probability"] <= 1 for p in preds)
    assert all(p["recommended_action"] for p in preds)

    sub_id = int(scores.sort("churn_probability", descending=True)["sub_id"][0])
    row = client.get(f"/v1/subscribers/{sub_id}").json()
    assert row["risk_band"] == "critical" and row["reasons"]
    assert client.get("/v1/subscribers/999999999").status_code == 404
    assert client.post("/v1/predict", json={"subscribers": []}).status_code == 422


def test_registry_gate_promote_and_rollback(small_cfg, trained):
    from churn import train
    from churn.registry import Registry, gate

    reg = Registry(small_cfg["paths"]["registry_dir"])
    first = reg.champion()
    assert first == trained["model_version"] and trained["promoted"]

    again = train.run(small_cfg)  # same data -> challenger ties the champion -> promoted
    assert again["promoted"] and reg.champion() == again["model_version"] != first
    assert reg.rollback() == first and reg.champion() == first

    ok, _ = gate({"pr_auc": 0.50, "capture_top10": 0.70}, {"pr_auc": 0.52, "capture_top10": 0.70}, 0.005)
    assert not ok  # PR-AUC regression beyond tolerance is blocked


def test_backtest_measures_realised_performance(small_cfg, trained):
    from churn import backtest, score

    score.run(small_cfg, "2025-09")
    report = backtest.run(small_cfg, "2025-09")
    assert report["realised"]["roc_auc"] > 0.75
    assert report["status"] in ("HEALTHY", "DEGRADED")
    assert report["campaign"] is not None and report["campaign"]["treated"] > 0
