"""Synthetic data warehouse for a prepaid-heavy mobile operator (Bangladesh-style market).

Real operator data cannot be published, so this module simulates the extracts a churn team
would pull from the DWH. It uses the same grain and columns, and churn is driven by behaviour
rather than random labels:

* ``subscribers``            one row per SIM: activation date, region, handset, plan, channels
* ``usage_monthly``          one row per subscriber-month *with activity* (voice, data, SMS, revenue,
                             active days, complaints, experienced call-drop rate, data speed ...)
* ``recharges``              one row per top-up transaction (timestamp, amount, channel)
* ``network_market``         one row per division-month: network KPIs + competitor promo flag

Each month, a subscriber's hazard of *starting to disengage* depends on
tenure (early-life churn), engagement, multi-SIM behaviour, accumulated dissatisfaction
(call drops, complaints), competitor promotions, a July tariff hike (price-sensitive users),
on-net community share, plan type and self-care app adoption. Disengaging subscribers
wind usage and recharges down over 0-3 months, then go silent. That silence is what the
60-day inactivity label picks up. Some recover, and some churn abruptly with no warning,
so the problem is realistically hard.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl

from churn.config import add_months, parse_month

DIVISIONS = ["Dhaka", "Chattogram", "Rajshahi", "Khulna", "Rangpur", "Mymensingh", "Sylhet", "Barishal"]
DIVISION_P = np.array([0.30, 0.20, 0.11, 0.10, 0.09, 0.08, 0.07, 0.05])
URBAN_P = np.array([0.65, 0.45, 0.30, 0.30, 0.22, 0.25, 0.30, 0.25])
DIVISION_DROP_BASE = np.array([0.011, 0.012, 0.013, 0.013, 0.015, 0.014, 0.017, 0.016])
HANDSETS = ["feature_phone", "smartphone_3g", "smartphone_4g"]
CHANNELS = ["retailer", "mfs", "app", "card"]
DENOMINATIONS = np.array([10, 20, 29, 30, 49, 50, 69, 99, 100, 149, 199, 249, 299, 399, 499, 699, 999])
EID_MONTHS = {3: 1.12, 6: 1.10}  # Eid-ul-Fitr / Eid-ul-Adha (2025) usage bump
TARIFF_HIKE_MONTH = 7  # new fiscal-year duty on mobile services takes effect in July


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _month_series(start, offsets: np.ndarray) -> pl.Series:
    idx = start.year * 12 + (start.month - 1) + offsets
    return (
        pl.select(pl.date(pl.Series(idx // 12), pl.Series(idx % 12 + 1), 1))
        .to_series()
        .alias("activation_month")
    )


def simulate(
    n_subscribers: int = 300_000,
    start_month: str = "2025-01",
    months: int = 12,
    gross_add_rate: float = 0.04,
    seed: int = 42,
) -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    start = parse_month(start_month)
    month_starts = [add_months(start, t - 1) for t in range(1, months + 1)]

    # ------------------------------------------------------------------ subscriber base
    n0 = n_subscribers
    adds = int(round(n0 * gross_add_rate))
    n = n0 + adds * months
    # activation month index: <= 0 for the opening base, 1..months for gross adds
    tenure0 = np.where(rng.random(n0) < 0.18, rng.integers(0, 12, n0), rng.integers(12, 160, n0))
    act = np.concatenate([-tenure0, np.repeat(np.arange(1, months + 1), adds)])

    division = rng.choice(len(DIVISIONS), n, p=DIVISION_P)
    urban = rng.random(n) < URBAN_P[division]
    age = np.clip(rng.normal(32, 10, n), 18, 75).astype(np.int16)
    gender = np.where(rng.random(n) < 0.62, "M", "F")
    postpaid = rng.random(n) < (0.02 + 0.05 * urban)
    handset_p = np.where(urban[:, None], [0.18, 0.12, 0.70], [0.42, 0.18, 0.40])
    handset = (rng.random(n)[:, None] > np.cumsum(handset_p, axis=1)).sum(axis=1)
    smartphone = handset > 0
    dual_sim = rng.random(n) < (0.45 + 0.12 * smartphone)
    app_user = smartphone & (rng.random(n) < 0.45)
    mfs_user = rng.random(n) < (0.50 + 0.15 * urban)

    # latent traits: never exported, only expressed through behaviour
    z_eng = rng.normal(0, 1, n)
    price_sens = rng.normal(0, 1, n) + 0.4 * ~urban
    arpu_scale = rng.lognormal(0, 0.45, n)
    onnet_pref = rng.beta(5, 4, n)
    frailty = rng.normal(0, 0.45, n)  # unobserved churn propensity

    # ------------------------------------------------------------ division-month context
    drop_base = DIVISION_DROP_BASE[:, None] * rng.lognormal(0, 0.12, (8, months + 1))
    outage = rng.random((8, months + 1)) < 0.06
    drop_base = drop_base + outage * rng.uniform(0.012, 0.03, (8, months + 1))
    promo = rng.random((8, months + 1)) < 0.10
    promo[[2, 4], 8:10] = True  # competitor data-pack blitz in Rajshahi & Rangpur, Aug-Sep

    # --------------------------------------------------------------------- monthly loop
    status = np.zeros(n, np.int8)  # 0 = not active yet, 1 = active, 2 = disengaging, 3 = churned
    dis_len = np.zeros(n, np.int8)
    dis_step = np.zeros(n, np.int8)
    dissat = np.zeros(n)
    usage_parts, recharge_parts, market_rows = [], [], []

    for t in range(1, months + 1):
        mstart = month_starts[t - 1]
        dim = (add_months(mstart, 1) - mstart).days
        status[(act <= t) & (status == 0)] = 1
        tenure = t - act
        live = (status == 1) | (status == 2)

        # experienced network quality and care contacts
        drop = drop_base[division, t] * (1 + 0.35 * ~urban) * rng.lognormal(0, 0.3, n)
        complaints = rng.poisson(0.03 + 12 * np.maximum(drop - 0.016, 0) + 0.06 * np.clip(dissat, 0, None))
        dissat = 0.6 * dissat + 60 * (drop - 0.016) + 0.6 * complaints + rng.normal(0, 0.3, n)

        promo_now = promo[division, t]
        hike = 1.0 if t in (TARIFF_HIKE_MONTH, TARIFF_HIKE_MONTH + 1) else 0.0
        logit = (
            -3.85
            + 1.7 * (tenure <= 2)
            + 0.6 * ((tenure > 2) & (tenure <= 12))
            - 0.35 * z_eng
            - 0.3 * np.log(arpu_scale)
            + 0.45 * dual_sim
            + 0.3 * np.clip(dissat, -2, 6)
            + promo_now * dual_sim * (0.55 + 0.35 * np.clip(price_sens, -1, 2))
            + hike * 0.45 * np.clip(price_sens, 0, None)
            - 1.3 * postpaid
            - 0.35 * app_user
            - 0.2 * mfs_user
            - 1.8 * (onnet_pref - 0.55)
            + 0.25 * (handset == 0)
            + frailty
        )

        # disengaging subscribers progress, may recover, or go silent
        dis = status == 2
        recover = dis & (rng.random(n) < 0.10)
        status[recover] = 1
        dis = status == 2
        dis_step[dis] += 1
        status[dis & (dis_step > dis_len)] = 3

        # new disengagement starts; ~25 % are abrupt (no wind-down)
        start_now = (status == 1) & (rng.random(n) < _sigmoid(logit))
        k = start_now.sum()
        lens = rng.choice([0, 1, 2, 3], k, p=[0.32, 0.25, 0.27, 0.16])
        idx = np.flatnonzero(start_now)
        dis_len[idx], dis_step[idx] = lens, 1
        status[idx] = np.where(lens == 0, 3, 2)

        live = (status == 1) | (status == 2)
        mult = np.where(status == 2, 1 - dis_step / (dis_len + 1.0), 1.0)
        rows = np.flatnonzero(live)
        # temporary dormancy (village trip, second SIM, travel abroad): looks like churn, is not
        dormant = (status[rows] == 1) & (
            rng.random(rows.size) < 0.035 + 0.02 * dual_sim[rows] + 0.02 * (t in EID_MONTHS)
        )
        m = np.where(dormant, rng.uniform(0.1, 0.6, rows.size), mult[rows])
        season = EID_MONTHS.get(mstart.month, 1.0)
        noise = rng.lognormal(0, 0.2, rows.size)
        hs = handset[rows]
        ze = z_eng[rows]
        la = np.log(arpu_scale[rows])
        shift = np.where(promo_now[rows] & dual_sim[rows] & (price_sens[rows] > 0.5), 0.75, 1.0)

        voice = np.exp(4.5 + 0.45 * ze + 0.35 * la + 0.4 * (hs == 0)) * noise * m * season
        voice *= rng.lognormal(0, 0.15, rows.size)
        onnet = voice * np.clip(onnet_pref[rows] + rng.normal(0, 0.05, rows.size), 0, 1)
        data_mu = np.select([hs == 0, hs == 1], [1.0 + 0.3 * ze, 6.3 + 0.6 * ze], 7.9 + 0.6 * ze + 0.3 * la)
        data_mb = np.exp(data_mu) * noise * m**1.4 * season * shift * rng.lognormal(0, 0.3, rows.size)
        speed_base = np.select([hs == 0, hs == 1], [np.nan, 3.0], 18.0)
        speed = speed_base / (1 + 40 * np.maximum(drop[rows] - 0.012, 0)) * rng.lognormal(0, 0.2, rows.size)
        sms = rng.poisson(8 * m * np.exp(0.2 * ze))
        packs = np.where(hs > 0, rng.poisson(data_mb / 900 + 0.2), 0)
        vas = rng.poisson(0.3, rows.size) * 10.0
        revenue = voice * 0.65 + data_mb / 1024 * 22 + sms * 0.4 + vas
        revenue *= 1.05 if t >= TARIFF_HIKE_MONTH else 1.0
        revenue = np.where(postpaid[rows], np.maximum(revenue, 499.0), revenue)
        loans = rng.poisson(0.2 + 0.25 * np.clip(price_sens[rows], 0, None) + 0.3 * (status[rows] == 2))
        loans = np.where(postpaid[rows], 0, loans)

        disen = (status[rows] == 2) | dormant
        # occasional users are active only a few days a month and are NOT churners
        p_day = _sigmoid(1.6 + 1.3 * ze + rng.normal(0, 0.4, rows.size))
        days_active = np.maximum(1, rng.binomial(dim, np.where(disen, p_day * np.sqrt(m), p_day))).astype(
            np.int16
        )
        gap_end = np.minimum(rng.geometric(np.clip(p_day, 0.05, 1)) - 1, dim - days_active)
        last_day = np.where(
            disen,
            np.minimum(dim, np.round(days_active / np.maximum(p_day, 0.05)) + rng.integers(0, 4, rows.size)),
            dim - gap_end,
        ).astype(np.int16)
        last_day = np.maximum(last_day, days_active)

        usage_parts.append(
            pl.DataFrame(
                {
                    "sub_id": rows.astype(np.int32),
                    "month": pl.Series([mstart] * rows.size, dtype=pl.Date),
                    "voice_min": voice.astype(np.float32),
                    "onnet_min": onnet.astype(np.float32),
                    "data_mb": data_mb.astype(np.float32),
                    "sms_cnt": sms.astype(np.int16),
                    "data_pack_cnt": packs.astype(np.int16),
                    "revenue": revenue.astype(np.float32),
                    "days_active": days_active,
                    "last_active_day": last_day,
                    "emergency_loan_cnt": loans.astype(np.int16),
                    "complaint_cnt": complaints[rows].astype(np.int16),
                    "drop_call_rate": drop[rows].astype(np.float32),
                    "data_speed_mbps": speed.astype(np.float32),
                }
            )
        )

        # ------------------------------------------------ recharge transactions (events)
        spend = revenue * rng.lognormal(0, 0.12, rows.size)
        lam = np.where(postpaid[rows], 1.0, spend / 55 * np.where(disen, 0.7, 1.0))
        cnt = rng.poisson(lam)
        cnt = np.where(~disen & ~postpaid[rows], np.maximum(cnt, 1), cnt)
        cnt = np.where(postpaid[rows], 1, cnt)
        ev_row = np.repeat(np.arange(rows.size), cnt)
        ev_sub = rows[ev_row]
        ev_day = 1 + np.floor(rng.random(ev_row.size) * last_day[ev_row]).astype(np.int64)
        ticket = spend[ev_row] / cnt[ev_row] * rng.lognormal(0, 0.35, ev_row.size)
        pos = np.clip(np.searchsorted(DENOMINATIONS, ticket), 0, len(DENOMINATIONS) - 1)
        lower = np.clip(pos - 1, 0, None)
        pick_lower = np.abs(DENOMINATIONS[lower] - ticket) < np.abs(DENOMINATIONS[pos] - ticket)
        amount = np.where(
            postpaid[ev_sub], np.round(spend[ev_row]), DENOMINATIONS[np.where(pick_lower, lower, pos)]
        )
        u = rng.random(ev_row.size)
        mfs_share = np.where(mfs_user[ev_sub], 0.45, 0.0)
        app_share = np.where(app_user[ev_sub], 0.20, 0.03)
        channel = np.where(
            u < mfs_share,
            1,
            np.where(u < mfs_share + app_share, 2, np.where(u < mfs_share + app_share + 0.05, 3, 0)),
        )
        recharge_parts.append(
            pl.DataFrame(
                {
                    "sub_id": ev_sub.astype(np.int32),
                    "day_offset": (ev_day - 1).astype(np.int32),
                    "minute": rng.integers(0, 1440, ev_row.size).astype(np.int32),
                    "month": pl.Series([mstart] * ev_row.size, dtype=pl.Date),
                    "amount": amount.astype(np.float32),
                    "channel": channel.astype(np.int8),
                }
            )
        )

        for d, name in enumerate(DIVISIONS):
            market_rows.append(
                {
                    "division": name,
                    "month": mstart,
                    "avg_drop_call_rate": float(drop_base[d, t] * (1 + 0.35 * (1 - URBAN_P[d]))),
                    "network_outage": bool(outage[d, t]),
                    "competitor_promo": bool(promo[d, t]),
                }
            )

    subscribers = pl.DataFrame(
        {
            "sub_id": np.arange(n, dtype=np.int32),
            "activation_month": _month_series(start, act - 1),
            "division": np.array(DIVISIONS)[division],
            "urban": urban,
            "gender": gender,
            "age": age,
            "plan_type": np.where(postpaid, "postpaid", "prepaid"),
            "handset": np.array(HANDSETS)[handset],
            "dual_sim": dual_sim,
            "app_user": app_user,
            "mfs_user": mfs_user,
        }
    )
    usage = pl.concat(usage_parts)
    recharges = (
        pl.concat(recharge_parts)
        .with_columns(
            ts=pl.col("month").cast(pl.Datetime("ms"))
            + pl.duration(days=pl.col("day_offset"), minutes=pl.col("minute"))
        )
        .with_columns(
            channel=pl.col("channel").replace_strict(dict(enumerate(CHANNELS)), return_dtype=pl.Utf8)
        )
        .select("sub_id", "ts", "amount", "channel")
        .sort("ts")
    )
    return {
        "subscribers": subscribers,
        "usage_monthly": usage,
        "recharges": recharges,
        "network_market": pl.DataFrame(market_rows),
    }


def write_tables(tables: dict[str, pl.DataFrame], raw_dir: str | Path) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.write_parquet(raw_dir / f"{name}.parquet", compression="zstd")


def load_tables(raw_dir: str | Path) -> dict[str, pl.DataFrame]:
    raw_dir = Path(raw_dir)
    names = ["subscribers", "usage_monthly", "recharges", "network_market"]
    missing = [n for n in names if not (raw_dir / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"Missing tables {missing} in {raw_dir}. Run `churn simulate` first.")
    return {n: pl.read_parquet(raw_dir / f"{n}.parquet") for n in names}


def run(cfg: dict) -> dict[str, pl.DataFrame]:
    sim = cfg["simulation"]
    t0 = time.perf_counter()
    tables = simulate(
        n_subscribers=sim["n_subscribers"],
        start_month=sim["start_month"],
        months=sim["months"],
        gross_add_rate=sim["gross_add_rate"],
        seed=cfg["seed"],
    )
    write_tables(tables, cfg["paths"]["raw_dir"])
    elapsed = time.perf_counter() - t0
    print(f"Simulated data warehouse in {elapsed:.1f}s -> {cfg['paths']['raw_dir']}")
    for name, df in tables.items():
        print(f"  {name:<16} {df.height:>12,} rows  {df.width:>3} cols")
    return tables
