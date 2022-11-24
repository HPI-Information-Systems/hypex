# HYPEX

A Framework for Hyperparameter Optimization in Time Series Anomaly Detection

## Development Setup

You need the following:

- Python 3.8 or Python 3.9
- Docker
- Every node must be able to connect to all other nodes via SSH and other sockets (ports)

Then you can execute:

```sh
git clone git@github.com:HPI-Information-Systems/hypex.git
cd hypex
make setup
```

This creates the dev environment in `.venv`, installs all dependencies, and HYPEX in development (editable) mode.

You can start HYPEX using the following command:

```sh
source .venv/bin/activate
python -m hypex -c example-config.yaml -e remote
```

## Update Dependencies

1. Add dependency to `requirements.in`, `requirements-ci.in`, or `requirements-dev.in`
2. Add dependency to `setup.cfg` under `install_requires` if its a runtime dependency
3. Pip-compile all dependencies to generate the `requirements*.txt`-files:

   ```sh
   make deps
   ```

4. To install the dependencies run `make install` or `make install-dev`.

## Release a New Version

1. Make sure to update `CHANGELOG.rst -> Unreleased`
2. Release via

   ```sh
   make release version=0.1.0
   ```
