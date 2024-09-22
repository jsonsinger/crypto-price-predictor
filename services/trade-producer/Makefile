run-dev:
	poetry run python -m src.main

build:
	docker build -t trade-producer .

run: build
	docker run \
		--network=redpanda_network \
		--env-file .env.prod \
		trade-producer

lint:
	poetry run ruff check --fix

format:
	poetry run ruff format .