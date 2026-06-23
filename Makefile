export UV_CACHE_DIR = /goinfre/$(USER)/.cache/uv
export HF_HOME = /goinfre/$(USER)/hf_cache
export TRANSFORMERS_CACHE = /goinfre/$(USER)/hf_cache/transformers
export TORCH_HOME = /goinfre/$(USER)/torch_cache

PYTHON = uv run python
MODULE = src
OUTPUT_DIR = data/output
CACHE_DIRS = .pytest_cache .mypy_cache .uv_cache

.PHONY: all run install debug clean lint lint-strict

all: run

run:
	$(PYTHON) -m $(MODULE)

install:
	uv sync

debug:
	$(PYTHON) -m pdb -m $(MODULE)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@for dir in $(CACHE_DIRS); do \
		if [ -d $$dir ]; then \
			rm -rf $$dir; \
		fi; \
	done
	@if [ -d $(OUTPUT_DIR) ]; then\
		rm -rf $(OUTPUT_DIR)/*; \
	fi

lint:
	uv run flake8 $(MODULE)
	uv run mypy $(MODULE) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 $(MODULE)
	uv run mypy $(MODULE) --strict
