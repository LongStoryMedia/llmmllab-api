PORT ?= 8000
DOCKER_IMAGE ?= llmmllab-api
DOCKER_TAG ?= latest
HELM_KUBECONTEXT ?= lsnet

export HELM_KUBECONTEXT

.SILENT

# Development
start:
	@echo "Starting API server on port $(PORT)..."
	@export $(shell grep -v '^#' .env 2>/dev/null | xargs 2>/dev/null) && \
	uv run python -m uvicorn app:app --host 0.0.0.0 --port $(PORT) --reload

start-docker:
	@echo "Starting API server in Docker..."
	@bash ./run.sh

# Testing
test:
	@echo "Running tests..."
	@uv run pytest test/

test-unit:
	@echo "Running unit tests..."
	@uv run pytest test/unit/

# Validation
validate:
	@echo "Validating Python syntax..."
	@python -m compileall -q -x '(venv|\.venv)' .
	@echo "Validation complete!"

# Cleanup
clean:
	@echo "Cleaning artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Artifacts cleaned."

# Docker
docker-build:
	@echo "Building Docker image $(DOCKER_IMAGE):$(DOCKER_TAG)..."
	@docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-push: docker-build
	@echo "Pushing Docker image..."
	@docker push $(DOCKER_IMAGE):$(DOCKER_TAG)

# Kubernetes
k8s-apply:
	@echo "Applying Kubernetes manifests..."
	@chmod +x k8s/apply.sh
	@DOCKER_TAG=$(DOCKER_TAG) ./k8s/apply.sh

deploy: docker-push k8s-apply
	@echo "Deployment complete!"

# Sync (k8s dev mode)
sync:
	@echo "Syncing code to k8s node..."
	@chmod +x sync-code.sh
	@./sync-code.sh

sync-watch:
	@echo "Watching for changes and syncing..."
	@chmod +x sync-code.sh
	@./sync-code.sh -w
