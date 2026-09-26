# Interview guide: defending this project

Short answers to the questions a telco data-science panel (Grameenphone, Robi, Banglalink, bKash-style fintechs) is likely to ask. Every answer points to the code that backs it.

## Problem framing

**How do you define churn for prepaid?**
Prepaid customers don't cancel; they stop using the SIM. A subscriber active in month T is labelled churn if there's no usage and no recharge in T+1 and T+2 (≈60 days). Operators typically use 60–90 days of no revenue-generating activity; the window is configurable (`churn.inactivity_months`). → `features.build_snapshot`

**Why "active in the cutoff month" as the population?**
You can only run retention on someone who is still reachable and still spending. Subscribers who already went silent are a win-back problem, not a retention problem.

**Why a 60-day forward window instead of next month?**
It gives the campaign team time to act, and it avoids counting temporary dormancy (Eid village trips, using the other SIM for a few weeks) as churn.

## Data and leakage

**How do you know there's no leakage?**
1. Features only read data ≤ the last day of the cutoff month.
2. A test deletes everything after the cutoff and asserts every feature is bit-identical (`test_features_do_not_leak_future_data`).
3. Evaluation is **out-of-time**: the test month (Oct) is after every training label window.

**Why not a random train/test split?**
Subscribers appear in several monthly snapshots, and behaviour shifts over time (tariff hike, competitor promos). A random split puts the same person and the same month in both sets and overstates performance. The model is used on *next* month, so it's evaluated on a future month.

**Why no SMOTE/ADASYN for the 4% churn rate?**
Gradient boosting handles imbalance fine for *ranking*. Resampling distorts probabilities, and the campaign maths (`p × value − cost`) needs calibrated probabilities. Brier = 0.024 and the calibration curve sits on the diagonal. v1 of this repo oversampled *before* splitting, which leaked synthetic copies of test rows into training.

**Is the data real?**
No. Subscriber data can't be published, so I simulated an operator DWH with realistic grain (monthly usage, event-level recharges, network/market tables) and a behavioural churn process with deliberate noise: abrupt churn, recovery, temporary dormancy. Results show the method; on real data I'd expect AUC in a similar 0.80–0.90 band.

## Model

**Why LightGBM and not deep learning?**
Tabular, heterogeneous features with missing values and monotone-ish effects are where boosted trees are strongest. It trains in about a minute on 1.5M rows, scores ~120k subscribers/s, and gives exact TreeSHAP explanations. The logistic-regression baseline shows the non-linearity is worth it (PR-AUC 0.55 vs 0.40).

**Which metric do you optimise?**
Ranking and business value, not accuracy. PR-AUC and capture@top-k show how many churners a fixed-size campaign reaches. KS is familiar to risk teams. The final judge is campaign **profit** on the test month.

**What are the top features?**
`days_since_last_activity`, `days_since_last_recharge`, voice/revenue/data **trends**, tenure, and the recharge-gap features. Recency and trend beat static profile; that matches practitioner experience in prepaid markets.

## Explainability and action

**How does the call centre know what to do?**
Per-subscriber TreeSHAP values are summed into 9 driver families. Symptoms (going quiet, spending less) are separated from root causes (early-life SIM, network, competitor offer, credit stress, multi-SIM). The action comes from the strongest root cause, because an offer should fix the cause, not the symptom. → `explain.primary_driver`

**Isn't SHAP slow?**
Yes, ~1000× a plain prediction. Batch runs explain only rows that get actioned (critical band + campaign targets): 1M subscribers score in 52 s end-to-end on 4 vCPUs. The API explains one subscriber in ~3 ms.

## Business

**How do you choose who to target?**
Expected value: `p(churn) × save_rate × margin × ARPU × 6 months − (contact + redeemed offer cost)`, targeting only where it's positive. On the test month that's 2.8% of the base for +201k ৳, within 0.4% of the hindsight optimum. Random targeting at the same size loses money. → `campaign.py`

**Where does `save_rate = 30%` come from?**
It's an assumption. In production I'd measure it with a randomised **hold-out control group** in every campaign, then move to an **uplift model**. Some high-risk customers leave regardless, and some would stay anyway. You want the *persuadable* ones.

## Production

**How is it deployed?**
A monthly batch job (`churn score`) writes scores and a campaign CSV for the CRM tool. A FastAPI service serves on-demand scores and subscriber lookups. The model artifact freezes feature order, category levels and risk thresholds, so inference never re-fits anything (another v1 bug).

**How do you know when to retrain?**
Every scoring run computes PSI per feature and for the score distribution against the validation month (> 0.25 → retrain). Once labels mature 60 days later, realised lift/capture is tracked against the test-month benchmark.

**What would you change at Grameenphone scale (80M+ subscribers)?**
Build snapshots in the warehouse (Spark/BigQuery/Hive) or a feature store instead of in-memory Polars. Partition scoring by region/sub_id range across workers (it's embarrassingly parallel). Keep LightGBM; it scales fine. Add call-graph (social network) features, because churn spreads through frequent contacts.
