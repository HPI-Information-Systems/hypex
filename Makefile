VENV_NAME?=.venv
PYTHON=${VENV_NAME}/bin/python3
PACKAGE_DIR=src/hypaad

setup:
	python3 -m venv ${VENV_NAME}
	${PYTHON} setup.py develop
	make install

setupCI:
	python3 -m venv ${VENV_NAME}
	${PYTHON} -m pip install -r requirements-ci.txt

clean:
	rm -r ${VENV_NAME}
	make setup

lint:
	${PYTHON} -m pylint ${PACKAGE_DIR}

format:
	${PYTHON} -m isort --profile black ${PACKAGE_DIR}
	${PYTHON} -m black -l 80 ${PACKAGE_DIR}

test:
	${PYTHON} -m pytest tests -vv

install:
	${PYTHON} -m pip install pip-tools
	${PYTHON} -m pip install -r requirements.txt

deps:
	${PYTHON} -m piptools compile requirements.in
	${PYTHON} -m piptools compile requirements-ci.in -o requirements-ci.txt
	make install

docs:
	${PYTHON} -m tox -e docs

package:
	${PYTHON} setup.py sdist

# make release version=0.1.0
release:
	@echo "Version: $(version)"
	make lint
	@sh ./sed -i 's/Unreleased/Unreleased\n==========\n\nv$(version)/g' CHANGELOG.rst
	git add CHANGELOG.rst && git commit -m "Release v$(version)" && git push
	git tag v$(version) && git push origin v$(version)
	@echo "Successfully released version v$(version)"