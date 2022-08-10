# NOTE: Used on linux, limited support outside of Linux
#
# A simple makefile to help with small tasks related to development of autosklearn
# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

.PHONY: help install-dev pre-commit clean clean-doc clean-build build doc links publish test

help:
	@echo "Makefile autosklearn"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* pre-commit       to run the pre-commit check"
	@echo "* doc              to generate and view the html files"
	@echo "* linkcheck        to check the documentation links"
	@echo "* publish          to help publish the current branch to pypi"
	@echo "* test             to run the tests"

PYTHON ?= python
CYTHON ?= cython
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

install-dev:
	$(PIP) install -e ".[test,docs]"
	pre-commit install

pre-commit:
	$(PRECOMMIT) run --all-files

clean-build:
	rm -rf ${BUILD}

clean-doc:
	$(MAKE) -C ${DOCDIR} clean

clean: clean-build clean-doc

build:
	python setup.py develop

# Running build before making docs is needed all be it very slow.
# Without doing a full build, the doctests seem to use docstrings from the last compiled build
doc: clean build
	$(MAKE) -C ${DOCDIR} html
	@echo
	@echo "View docs at:"
	@echo ${INDEX_HTML}

links:
	$(MAKE) -C ${DOCDIR} linkcheck

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
	@echo "        pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ autosklearn"
	@echo
	@echo "* Run this to make sure it can import correctly, plus whatever else you'd like to test:"
	@echo
	@echo "        python -c 'import autosklearn'"
	@echo
	@echo "Once you have decided it works, publish to actual pypi with"
	@echo
	@echo "    python -m twine upload dist/*"

test:
	$(PYTEST) test
