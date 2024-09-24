IMAGE_NAME=trade-producer
KAFKA_TOPIC=trades_backfill

run-dev:
	poetry run python -m src.main

run-dev-backfill:
	BACKFILL_MODE=true \
	KAFKA_TOPIC=$(KAFKA_TOPIC) \
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
		-e BACKFILL_MODE=true \
		-e KAFKA_TOPIC=$(KAFKA_TOPIC) \
		$(IMAGE_NAME)

clean:
	docker rmi $(IMAGE_NAME)

lint:
	poetry run ruff check --fix

format:
	poetry run ruff format .