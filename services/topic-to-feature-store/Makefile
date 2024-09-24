IMAGE_NAME=topic-to-feature-store
BATCH_SIZE=3000
BACKFILL_INPUT_TOPIC=ohlcv_backfill
BACKFILL_CONSUMER_GROUP=backfill_consumer_feature_store

run-dev:
	poetry run python -m src.main

run-dev-backfill:
	BATCH_SIZE=$(BATCH_SIZE) \
	KAFKA_INPUT_TOPIC=$(BACKFILL_INPUT_TOPIC) \
	KAFKA_CONSUMER_GROUP=$(BACKFILL_CONSUMER_GROUP) \
	poetry run python -m src.main

# --no-cache flag is used to ensure that the latest version of the code is used e.g. docker build --no-cache -t topic-to-feature-store .
build:
	docker build -t $(IMAGE_NAME) .

run: build
	docker run \
		--network=redpanda_network \
		--env-file .env.prod \
		--env-file credentials.env \
		$(IMAGE_NAME)

run-backfill: build
	docker run \
		--network=redpanda_network \
		--env-file .env.prod \
		-e BATCH_SIZE=$(BATCH_SIZE) \
		-e KAFKA_INPUT_TOPIC=$(BACKFILL_INPUT_TOPIC) \
		-e KAFKA_CONSUMER_GROUP=$(BACKFILL_CONSUMER_GROUP) \
		$(IMAGE_NAME)

clean:
	docker rmi $(IMAGE_NAME)

lint:
	poetry run ruff check --fix

format:
	poetry run ruff format .