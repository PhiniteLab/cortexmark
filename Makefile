.PHONY: all convert clean chunk render analyze validate lint format typecheck test clean-outputs clean-local-artifacts help

PYTHON ?= $(shell if [ -x .venv/bin/python ]; then printf %s .venv/bin/python; elif command -v python3 >/dev/null 2>&1; then command -v python3; else command -v python; fi)
ifneq ($(origin CONFIG), undefined)
export CONFIG
endif
CONFIG ?= configs/pipeline.yaml

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

all: ## Run the full pipeline (convert → clean → chunk → render)
	$(PYTHON) -m cortexmark.run_pipeline --config $(CONFIG)

convert: ## PDF → Markdown (docling + markitdown)
	$(PYTHON) -m cortexmark.convert --config $(CONFIG)

clean: ## Clean raw Markdown
	$(PYTHON) -m cortexmark.clean --config $(CONFIG)

chunk: ## Split cleaned Markdown into chunks
	$(PYTHON) -m cortexmark.chunk --config $(CONFIG)

render: ## Populate source templates
	$(PYTHON) -m cortexmark.render_templates --config $(CONFIG)

analyze: ## Run analysis modules (semantic chunk, cross-ref, algorithm, notation)
	$(PYTHON) -m cortexmark.run_pipeline --config $(CONFIG) --stages analyze

validate: ## Run validation modules (formula, scientific QA, citation context)
	$(PYTHON) -m cortexmark.run_pipeline --config $(CONFIG) --stages validate

lint: ## Run ruff linter + formatter check
	$(PYTHON) -m ruff check cortexmark/ tests/
	$(PYTHON) -m ruff format --check cortexmark/ tests/

format: ## Auto-format code with ruff
	$(PYTHON) -m ruff format cortexmark/ tests/
	$(PYTHON) -m ruff check --fix cortexmark/ tests/

typecheck: ## Run pyright type checker
	$(PYTHON) -m pyright cortexmark/

test: ## Run tests with pytest + coverage
	$(PYTHON) -m pytest tests/ -v

clean-outputs: ## Remove configured outputs inside this project (use with caution)
	$(PYTHON) -m cortexmark.cleanup outputs

clean-local-artifacts: ## Remove caches/build artifacts, preserving configured data/results
	$(PYTHON) -m cortexmark.cleanup local-artifacts
