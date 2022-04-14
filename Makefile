VENV_NAME?=.venv
PYTHON=${VENV_NAME}/bin/python3
PACKAGE_DIR=src/hypaad

cluster:
	docker compose down -v --remove-orphans
	docker build --ssh default . -t hypaad-controller -f Dockerfile.controller
	docker build --ssh default . -t hypaad-node -f Dockerfile.node
	docker-compose up --build

run:
	make package
	make push-images
	docker exec hypaad_controller /bin/sh -c "ssh-keyscan node-0 >> ~/.ssh/known_hosts"
	docker exec hypaad_controller /bin/sh -c "ssh-keyscan node-1 >> ~/.ssh/known_hosts"
	docker exec hypaad_controller /bin/sh -c "python3 ${PACKAGE_DIR}"

logs:
	docker compose logs -f

run-it:
	docker exec -it hypaad_controller /bin/sh
run-it0:
	docker exec -it hypaad_node-0 /bin/sh
run-it1:
	docker exec -it hypaad_node-1 /bin/sh

push-images:
	docker image tag sopedu:5000/akita/series2graph localhost:5000/akita/series2graph
	docker push localhost:5000/akita/series2graph

setup:
	python3 -m venv ${VENV_NAME}
	${PYTHON} -m pip install --upgrade pip
	${PYTHON} setup.py develop
	make install

setupCI:
	python3 -m venv ${VENV_NAME}
	${PYTHON} -m pip install --upgrade pip
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
	${PYTHON} -m pip install -r requirements-dev.txt

deps:
	${PYTHON} -m piptools compile requirements.in
	${PYTHON} -m piptools compile requirements-base.in -o requirements-base.txt
	${PYTHON} -m piptools compile requirements-dev.in -o requirements-dev.txt
	${PYTHON} -m piptools compile requirements-ci.in -o requirements-ci.txt
	make install

docs:
	${PYTHON} -m tox -e docs

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
