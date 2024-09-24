run-dev:
	poetry run python -m src.main

run-dev-backfill:
	KAFKA_INPUT_TOPIC=trades_backfill \
	KAFKA_OUTPUT_TOPIC=ohlcv_backfill \
	KAFKA_CONSUMER_GROUP=trade_to_ohlcv_backfill \
	poetry run python -m src.main

build:
	docker build -t trade-to-ohlcv .

run: build
	docker run \
		--network=redpanda_network \
		--env-file .env.prod \
		trade-to-ohlcv

run-backfill: build
	docker run \
		--network=redpanda_network \
		--env-file .env.prod \
		-e KAFKA_INPUT_TOPIC=trades_backfill \
		-e KAFKA_OUTPUT_TOPIC=ohlcv_backfill \
		-e KAFKA_CONSUMER_GROUP=trade_to_ohlcv_backfill \
		trade-to-ohlcv

lint:
	poetry run ruff check --fix

format:
	poetry run ruff format .