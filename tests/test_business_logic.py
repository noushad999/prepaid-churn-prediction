import numpy as np
import pandas as pd
import pytest

from churn.campaign import campaign_summary, contact_cost, expected_net, profit_curve
from churn.evaluate import classification_metrics, gains_table
from churn.explain import SYMPTOMS, family_matrix, feature_family, primary_driver
from churn.monitor import build_reference, drift_report

CAMPAIGN = {
    "contact_cost": 2.0,
    "offer_cost": 40.0,
    "offer_redeem_rate": 0.6,
    "save_rate": 0.3,
    "margin": 0.55,
    "clv_months": 6,
}


def test_expected_net_breakeven():
    assert contact_cost(CAMPAIGN) == pytest.approx(26.0)
    arpu = np.array([200.0])
    breakeven_p = contact_cost(CAMPAIGN) / (0.3 * 0.55 * 200 * 6)
    assert expected_net(np.array([breakeven_p]), arpu, CAMPAIGN)[0] == pytest.approx(0.0)
    assert (
        expected_net(np.array([0.9]), arpu, CAMPAIGN)[0]
        > 0
        > expected_net(np.array([0.01]), arpu, CAMPAIGN)[0]
    )


def test_perfect_ranking_beats_random():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    arpu = rng.lognormal(5, 0.5, 5000)
    perfect = profit_curve(y, y + rng.random(5000) * 0.1, arpu, CAMPAIGN).max()
    random = profit_curve(y, rng.random(5000), arpu, CAMPAIGN).max()
    assert perfect > random
    summary = campaign_summary(y, y * 0.9 + 0.01, arpu, CAMPAIGN)
    assert summary["policy_profit"] > summary["random_targeting_profit_same_k"]


def test_metrics_on_perfect_scores():
    y = np.array([0] * 90 + [1] * 10)
    m = classification_metrics(y, y.astype(float))
    assert m["roc_auc"] == 1.0 and m["capture_top10"] == 1.0 and m["lift_top10"] == pytest.approx(10.0)
    g = gains_table(y, y.astype(float))
    assert g["cum_capture"].iloc[-1] == pytest.approx(1.0) and len(g) == 10


def test_feature_families():
    assert feature_family("rch_max_gap_days_90d") == "recharge_slowdown"
    assert feature_family("days_since_last_recharge") == "recharge_slowdown"
    assert feature_family("days_since_last_activity") == "inactivity"
    assert feature_family("drop_call_rate_m0") == "network_experience"
    assert feature_family("tenure_months") == "early_life"
    assert feature_family("age") == "profile"
    fams, F = family_matrix(["tenure_months", "age", "data_mb_m0"])
    assert F.sum(axis=1).tolist() == [1, 1, 1] and len(fams) == F.shape[1]


def test_primary_driver_prefers_root_cause_over_symptom():
    fams = ["inactivity", "network_experience", "early_life"]
    scores = np.array([[2.0, 0.4, 0.1], [2.0, 0.01, 0.0]])
    drivers = primary_driver(scores, fams)
    assert drivers[0] == "network_experience"  # cause wins even though the symptom is larger
    assert drivers[1] == "inactivity"  # no meaningful cause -> fall back to the symptom
    assert "inactivity" in SYMPTOMS


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    ref_X = pd.DataFrame({"x": rng.normal(0, 1, 20000), "c": pd.Categorical(rng.choice(["a", "b"], 20000))})
    ref = build_reference(ref_X, rng.random(20000))
    same = drift_report(ref, ref_X, rng.random(20000))
    assert same["score_status"] == "stable" and same["n_major"] == 0
    shifted_X = pd.DataFrame({"x": rng.normal(1.5, 1, 20000), "c": pd.Categorical(["a"] * 20000)})
    shifted = drift_report(ref, shifted_X, rng.random(20000) ** 3)
    assert shifted["n_major"] == 2 and shifted["score_status"] == "major"
