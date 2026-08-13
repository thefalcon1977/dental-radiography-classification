# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SOURCEDIR     = .
BUILDDIR      = _build

# Prefer project venv when present (avoids "sphinx-build: command not found")
VENV_SPHINX   := $(abspath $(CURDIR)/../.venv/bin/sphinx-build)
ifeq ($(wildcard $(VENV_SPHINX)),)
  SPHINXBUILD ?= sphinx-build
else
  SPHINXBUILD ?= $(VENV_SPHINX)
endif

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
