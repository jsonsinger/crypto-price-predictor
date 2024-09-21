IMAGE_NAME=topic-to-feature-store

run-dev:
	python -m src.main

build:
	docker build -t $(IMAGE_NAME) .

run: build
	docker run \
		--network=redpanda_network \
		--env-file .env.prod \
		--env-file credentials.env \
		$(IMAGE_NAME)

clean:
	docker rmi $(IMAGE_NAME)

lint:
	poetry run ruff check --fix

format:
	poetry run ruff format .