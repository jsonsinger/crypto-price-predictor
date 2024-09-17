start-redpanda:
	@echo "Starting Redpanda..."
	docker compose -f redpanda.yml up -d