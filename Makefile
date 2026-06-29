INSTALL_DIR := $(HOME)/.local/bin
REPO_DIR    := $(shell cd "$(dir $(lastword $(MAKEFILE_LIST)))" && pwd)
UV          := uv
VENV        := $(REPO_DIR)/.venv
COMMAND     := demucs-mlx

.PHONY: install uninstall deps venv help

help: ## Show this help
	@echo ""
	@echo "  demucs-mlx installer"
	@echo "  ────────────────────"
	@echo ""
	@echo "  make install     Create .venv, install demucs-mlx, add it to PATH"
	@echo "  make uninstall   Remove demucs-mlx from your PATH"
	@echo "  make deps        Create .venv and install demucs-mlx (editable)"
	@echo "  make venv        Create the uv virtual environment (.venv) only"
	@echo ""
	@echo "  Or, for a one-shot user install:  uv tool install ."
	@echo ""

venv: ## Create the uv virtual environment (.venv) if missing
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating uv virtual environment at $(VENV)..."; \
		$(UV) venv "$(VENV)"; \
	fi

deps: venv ## Install demucs-mlx (editable) and dependencies into .venv
	@echo "Installing demucs-mlx and dependencies..."
	@$(UV) pip install --python "$(VENV)" -e "$(REPO_DIR)" -q
	@echo "Done."

install: deps ## Install demucs-mlx into ~/.local/bin
	@mkdir -p "$(INSTALL_DIR)"
	@echo '#!/bin/bash' > "$(INSTALL_DIR)/$(COMMAND)"
	@echo '# demucs-mlx — Music source separation on Apple Silicon' >> "$(INSTALL_DIR)/$(COMMAND)"
	@echo 'exec "$(VENV)/bin/python" -m demucs_mlx "$$@"' >> "$(INSTALL_DIR)/$(COMMAND)"
	@chmod +x "$(INSTALL_DIR)/$(COMMAND)"
	@echo ""
	@echo "  ✓ Installed $(COMMAND) → $(INSTALL_DIR)/$(COMMAND)"
	@echo ""
	@if echo "$$PATH" | tr ':' '\n' | grep -qx "$(INSTALL_DIR)"; then \
		echo "  Ready! Try: $(COMMAND) --help"; \
	else \
		echo "  ⚠ $(INSTALL_DIR) is not in your PATH."; \
		echo ""; \
		echo "  Add this line to your shell profile (~/.zshrc or ~/.bashrc):"; \
		echo ""; \
		echo "    export PATH=\"$(INSTALL_DIR):\$$PATH\""; \
		echo ""; \
		echo "  Then restart your terminal or run: source ~/.zshrc"; \
	fi
	@echo ""

uninstall: ## Remove demucs-mlx from PATH
	@rm -f "$(INSTALL_DIR)/$(COMMAND)"
	@echo "  ✓ Removed $(COMMAND) from $(INSTALL_DIR)"
