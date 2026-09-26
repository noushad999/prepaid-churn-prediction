"""Command line entry point: `churn <command>` (or `python -m churn <command>`)."""

from __future__ import annotations

import argparse
import time


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="churn", description="Telco prepaid churn prediction pipeline")
    parser.add_argument("--config", default=None, help="Path to config YAML (default: configs/config.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("simulate", help="Generate the synthetic operator data warehouse")
    p_sim.add_argument("--subscribers", type=int, default=None, help="Initial active base size")
    sub.add_parser("train", help="Build snapshots, train + evaluate, export model and reports")
    p_score = sub.add_parser("score", help="Batch-score the active base for a month")
    p_score.add_argument(
        "--month", default=None, help="Snapshot month YYYY-MM (default: training.score_cutoff)"
    )
    p_all = sub.add_parser("pipeline", help="simulate -> train -> score")
    p_all.add_argument("--subscribers", type=int, default=None)
    p_serve = sub.add_parser("serve", help="Start the scoring API")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--scores", default=None, help="Batch scores parquet for /v1/subscribers lookups")

    args = parser.parse_args(argv)

    from churn.config import load_config

    overrides = {}
    if getattr(args, "subscribers", None):
        overrides = {"simulation": {"n_subscribers": args.subscribers}}
    cfg = load_config(args.config, overrides)
    t0 = time.perf_counter()

    if args.command in ("simulate", "pipeline"):
        from churn import simulate

        simulate.run(cfg)
    if args.command in ("train", "pipeline"):
        from churn import train

        train.run(cfg)
    if args.command in ("score", "pipeline"):
        from churn import score

        score.run(cfg, getattr(args, "month", None))
    if args.command == "serve":
        import os
        from pathlib import Path

        import uvicorn

        from churn.api import create_app

        scores = args.scores or os.getenv("CHURN_SCORES")
        if scores is None:
            candidates = sorted(Path(cfg["paths"]["scores_dir"]).glob("scores_*.parquet"))
            scores = str(candidates[-1]) if candidates else None
        uvicorn.run(create_app(cfg["paths"]["model_dir"], scores), host=args.host, port=args.port)
        return
    print(f"\nDone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
