training:
	poetry run python -m src.training

build:
	docker build -t price_predictor_training .

run: build
	docker run --env-file .env price_predictor_training

lint:
	poetry run ruff check --fix

format:
	poetry run ruff format .