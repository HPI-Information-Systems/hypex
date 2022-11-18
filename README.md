# HYPEX

A Framework for Hyperparameter Optimization in Time Series Anomaly Detection

## Development Setup

Creates dev environment in `.venv` and installs all dependencies.

```sh
make setup
```

## Local Execution

1. Start a local cluster using

   ```sh
   make cluster
   ```

2. Run the parameter optimization on the local cluster using

   ```sh
   make run
   ```

## Update Dependencies

1. Add dependency to `requirements.in`
2. Add dependency to `setup.cfg` under `install_requires`
3. Pip-compile dependencies

   ```sh
   make deps
   ```

   This will also install the new dependencies.

## Release a New Version

1. Make sure to update `CHANGELOG.rst -> Unreleased`
2. Release via

   ```sh
   make release version=0.1.0
   ```
