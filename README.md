# HYPAAD
> A Framework for Hyper-Parameter Optimization in Anomaly Detection

## Development Setup
Creates dev environment in `.venv` and installs all dependencies.

```bash
make setup
```

## Local Execution
1. Start a local cluster using
   ```
   make cluster
   ```
2. Run the parameter optimization on the local cluster using
   ```
   make run
   ```

## Update Dependencies

1. Add dependency to `requirements.in`
1. Add dependency to `setup.cfg` under `install_requires`
1. Pip-compile dependencies
   ```
   make deps
   ```
   This will also install the new dependencies.

## Release a New Version

1. Make sure to update `CHANGELOG.rst -> Unreleased`
1. Release via
   ```
   make release version=0.1.0
   ```
