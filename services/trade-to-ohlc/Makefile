IMAGE_NAME=trade-to-ohlcv

run-dev:
	poetry run python -m src.main

run-dev-backfill:
	KAFKA_INPUT_TOPIC=trades_backfill \
	KAFKA_OUTPUT_TOPIC=ohlcv_backfill \
	KAFKA_CONSUMER_GROUP=trade_to_ohlcv_backfill \
	AUTO_OFFSET_RESET=earliest \
	poetry run python -m src.main

build:
	docker build -t $(IMAGE_NAME) .

run: build
	docker run \
		--network=redpanda_network \
		--env-file .env.prod \
		$(IMAGE_NAME)

run-backfill: build
	docker run \
		--network=redpanda_network \
		--env-file .env.prod \
		-e KAFKA_INPUT_TOPIC=trades_backfill \
		-e KAFKA_OUTPUT_TOPIC=ohlcv_backfill \
		-e KAFKA_CONSUMER_GROUP=trade_to_ohlcv_backfill \
		$(IMAGE_NAME)

clean:
	docker rmi $(IMAGE_NAME)

lint:
	poetry run ruff check --fix

format:
	poetry run ruff format .