run-dev:
	poetry run python src/main.py

build:
	docker build -t trade-to-ohlcv .

run: build
	docker run \
		--network=redpanda_network \
		--env-file .env.prod \
		trade-to-ohlcv

lint:
	poetry run ruff check --fix

format:
	poetry run ruff format .