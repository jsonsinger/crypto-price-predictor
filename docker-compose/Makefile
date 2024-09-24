start-redpanda:
	@echo "Starting Redpanda..."
	docker compose -f redpanda.yml up -d

stop-redpanda:
	@echo "Stopping Redpanda..."
	docker compose -f redpanda.yml down

start-feature-pipeline:
	@echo "Starting Feature Pipeline..."
	docker compose -f feature-pipeline.yml up --build -d

stop-feature-pipeline:
	@echo "Stopping Feature Pipeline..."
	docker compose -f feature-pipeline.yml down

start-backfill-pipeline:
	@echo "Starting Backfill Pipeline..."
	docker compose -f backfill-pipeline.yml up --build -d

stop-backfill-pipeline:
	@echo "Stopping Backfill Pipeline..."
	docker compose -f backfill-pipeline.yml down