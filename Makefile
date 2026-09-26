PY ?= python
SUBSCRIBERS ?= 300000

.PHONY: install simulate train score pipeline serve test lint docker

install:        ## install package + dev tools
	$(PY) -m pip install -e ".[dev]"

simulate:       ## generate the synthetic operator data warehouse
	churn simulate --subscribers $(SUBSCRIBERS)

train:          ## build snapshots, train, evaluate, export model + reports
	churn train

score:          ## batch-score the latest month, build campaign list, check drift
	churn score

pipeline:       ## simulate -> train -> score
	churn pipeline --subscribers $(SUBSCRIBERS)

serve:          ## run the scoring API on :8000
	churn serve

test:
	$(PY) -m pytest

lint:
	ruff check src tests && ruff format --check src tests

docker:
	docker build -t telco-churn:latest .
	docker run --rm -p 8000:8000 -v $(PWD)/artifacts:/app/artifacts -v $(PWD)/outputs:/app/outputs \
		-e CHURN_SCORES=/app/outputs/scores_2025-12.parquet telco-churn:latest
