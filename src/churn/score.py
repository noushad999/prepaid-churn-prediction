"""Monthly batch scoring of the whole active base, with reason codes, campaign selection and drift check."""

from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl

from churn.campaign import expected_net
from churn.features import build_snapshot
from churn.model import ChurnModel
from churn.monitor import drift_report, load_reference
from churn.simulate import load_tables


def score_snapshot(
    model: ChurnModel,
    snap: pl.DataFrame,
    campaign_cfg: dict,
    explain_from_band: str = "critical",
    chunk_size: int = 250_000,
) -> pl.DataFrame:
    """Score every subscriber; explain (reason codes) those at/above `explain_from_band` or worth targeting."""
    arpu = snap["revenue_avg3m"].to_numpy()
    parts = []
    for start in range(0, snap.height, chunk_size):
        X = model.matrix(snap.slice(start, chunk_size))
        p = model.predict(X)
        ev_chunk = expected_net(p, arpu[start : start + len(p)], campaign_cfg)
        explain = (p >= model.metadata["risk_thresholds"][explain_from_band]) | (ev_chunk > 0)
        out = model.score_frame(X, explain=explain, p=p)
        parts.append(
            pl.DataFrame(
                {
                    k: pl.Series(k, v.tolist(), dtype=pl.Utf8) if v.dtype == object else v
                    for k, v in out.items()
                }
            )
        )
    scored = pl.concat(parts)
    ev = expected_net(scored["churn_probability"].to_numpy(), arpu, campaign_cfg)
    out = (
        snap.select(
            "sub_id",
            "snapshot",
            "division",
            "plan_type",
            "handset",
            "tenure_months",
            pl.col("revenue_avg3m").alias("arpu_3m"),
        )
        .hstack(scored)
        .with_columns(
            expected_net_value=pl.Series(ev, dtype=pl.Float32),
            target=pl.Series(ev > 0),
            churn_probability=pl.col("churn_probability").cast(pl.Float32),
        )
    )
    return out.with_columns(
        risk_rank=pl.col("churn_probability").rank("ordinal", descending=True).cast(pl.Int32)
    )


def run(cfg: dict, month: str | None = None) -> pl.DataFrame:
    paths = cfg["paths"]
    month = month or cfg["training"]["score_cutoff"]
    out_dir = Path(paths["scores_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = Path(paths["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model = ChurnModel.load(paths["model_dir"])
    tables = load_tables(paths["raw_dir"])
    t_load = time.perf_counter()
    snap = build_snapshot(tables, month, lookback_months=cfg["churn"]["lookback_months"], with_label=False)
    t_feat = time.perf_counter()
    scores = score_snapshot(model, snap, cfg["campaign"])
    t_score = time.perf_counter()

    scores.write_parquet(out_dir / f"scores_{month}.parquet")
    campaign = scores.filter(pl.col("target")).sort("expected_net_value", descending=True)
    campaign.drop("snapshot", "target").write_csv(out_dir / f"campaign_{month}.csv", float_precision=4)

    drift = drift_report(
        load_reference(paths["model_dir"]), model.frame(snap), scores["churn_probability"].to_numpy()
    )
    (reports / f"drift_{month}.json").write_text(json.dumps(drift, indent=2))

    n = scores.height
    print(f"Scored {n:,} active subscribers for {month}")
    print(
        f"  load {t_load - t0:.1f}s | features {t_feat - t_load:.1f}s | "
        f"score + SHAP reasons for critical/targets {t_score - t_feat:.1f}s ({n / (t_score - t_feat):,.0f} subs/s)"
    )
    bands = scores.group_by("risk_band").agg(
        pl.len().alias("subs"), pl.col("churn_probability").mean().alias("avg_p")
    )
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    for row in sorted(bands.iter_rows(named=True), key=lambda r: order[r["risk_band"]]):
        print(f"  {row['risk_band']:<9}{row['subs']:>9,} subs   avg p(churn) {row['avg_p']:.1%}")
    reasons = campaign.group_by("primary_driver").len().sort("len", descending=True)
    print(
        f"  campaign list: {campaign.height:,} subscribers, expected net value "
        f"{campaign['expected_net_value'].sum():,.0f} {cfg['campaign']['currency']}"
    )
    for row in reasons.iter_rows(named=True):
        print(f"    {str(row['primary_driver']):<24}{row['len']:>8,}")
    print(
        f"  drift: score PSI {drift['score_psi']:.3f} ({drift['score_status']}), "
        f"{drift['n_major']} major / {drift['n_moderate']} moderate feature shifts"
    )
    for r in drift["features"][:3]:
        print(f"    {r['feature']:<32} PSI {r['psi']:.3f} {r['status']}")
    print(f"  -> {out_dir / f'scores_{month}.parquet'}, {out_dir / f'campaign_{month}.csv'}")
    return scores
