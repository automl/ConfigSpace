# NOTE: Used on linux, limited support outside of Linux
#
# A simple makefile to help with small tasks related to development of ConfigSpace
# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

.PHONY: help install-dev install-test install-docs pre-commit clean clean-doc clean-build build docs links publish test clean-test

help:
	@echo "Makefile ConfigSpace"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* check            to run the pre-commit check"
	@echo "* docs             to generate and view the html files"
	@echo "* publish          to help publish the current branch to pypi"
	@echo "* test             to run the tests"

PYTHON ?= python
PYTEST ?= python -m pytest
CTAGS ?= ctags
PRECOMMIT ?= pre-commit
PIP ?= python -m pip
MAKE ?= make

DIR := "${CURDIR}"
DIST := "${DIR}/dist""
DOCDIR := "${DIR}/docs"
BUILD := "${DIR}/build"
INDEX_HTML := "file://${DOCDIR}/build/html/index.html"
NUMPY_INCLUDE := $(shell python -c 'import numpy; print(numpy.get_include())')

# https://stackoverflow.com/questions/40750596/how-do-i-escape-bracket-in-makefile
CP := )

benchmark:
	python scripts/benchmark_sampling.py

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

check:
	$(PRECOMMIT) run --all-files

fix:
	black --quiet ConfigSpace test
	ruff --silent --exit-zero --no-cache --fix ConfigSpace test

test:
	$(PYTEST) test

# Running build before making docs is needed all be it very slow.
# Without doing a full build, the doctests seem to use docstrings from the last compiled build
docs: clean build
	mkdocs serve

# Publish to testpypi
# Will echo the commands to actually publish to be run to publish to actual PyPi
# This is done to prevent accidental publishing but provide the same conveniences
publish:
	$(PIP) install twine
	$(PYTHON) -m twine upload --repository testpypi ${DIST}/*
	@echo
	@echo "Test with the following:"
	@echo "* Create a new virtual environment to install the uplaoded distribution into"
	@echo "* Run the following:"
	@echo
	@echo "        pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ConfigSpace"
	@echo
	@echo "* Run this to make sure it can import correctly, plus whatever else you'd like to test:"
	@echo
	@echo "        python -c 'import ConfigSpace'"
	@echo
	@echo "Once you have decided it works, publish to actual pypi with"
	@echo
	@echo "    python -m twine upload dist/*"
