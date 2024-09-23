run-dev:
	poetry run python -m src.main

run-dev-backfill:
	BACKFILL_TRADES=true \
	poetry run python -m src.main

build:
	docker build -t trade-producer .

run: build
	docker run \
		--network=redpanda_network \
		--env-file .env.prod \
		trade-producer

run-backfill: build
	docker run \
		--network=redpanda_network \
		--env-file .env.prod \
		-e BACKFILL_TRADES=true \
		trade-producer

lint:
	poetry run ruff check --fix

format:
	poetry run ruff format .