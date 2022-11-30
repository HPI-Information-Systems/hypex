VENV_NAME?=.venv
PYTHON=${VENV_NAME}/bin/python3
PACKAGE_DIR=src/hypaad
TEST_DIR=tests

.PHONY: release deps install-dev install-ci install test format lint clean setupCI setup create-env run-remote

run-remote:
	make package
	${PYTHON} ${PACKAGE_DIR} --environment remote --config evaluation.yaml 2>&1 | tee run.log

create-env:
	python3 -m venv ${VENV_NAME}
	${PYTHON} -m pip install --upgrade pip

setup: create-env install-dev

setupCI: create-env install-ci

clean:
	rm -r ${VENV_NAME}
	rm -r dist build
	rm -r **/__pycache__

lint:
	${PYTHON} -m pylint ${PACKAGE_DIR} --errors-only --fail-under 9

format:
	${PYTHON} -m isort --profile black ${PACKAGE_DIR}
	${PYTHON} -m black ${PACKAGE_DIR}

test:
	${PYTHON} -m pytest ${TEST_DIR} -vv

install:
	${PYTHON} -m pip install -r requirements.txt
	${PYTHON} -m pip install .

install-ci:
	${PYTHON} -m pip install -r requirements-ci.txt
	${PYTHON} -m pip install .

install-dev:
	${PYTHON} -m pip install -r requirements-dev.txt
	${PYTHON} -m pip install -e .

deps:
	${PYTHON} -m pip install pip-tools
	${PYTHON} -m piptools compile requirements.in
	${PYTHON} -m piptools compile requirements-ci.in -o requirements-ci.txt
	${PYTHON} -m piptools compile requirements-dev.in -o requirements-dev.txt

package:
	rm -f dist/*
	${PYTHON} setup.py sdist bdist_egg

# make release version=0.1.0
release:
	@echo "Version: $(version)"
	make lint
	@sh ./sed -i 's/Unreleased/Unreleased\n==========\n\nv$(version)/g' CHANGELOG.rst
	git add CHANGELOG.rst && git commit -m "Release v$(version)" && git push
	git tag v$(version) && git push origin v$(version)
	@echo "Successfully released version v$(version)"
