## Cursor Cloud specific instructions

### Overview

**scopyon** is a Python scientific computing library (Monte Carlo simulation toolkit for bioimaging systems). It is a pure Python library with no external services, databases, or Docker dependencies.

### Prerequisites

- **Python >= 3.13** (installed via `uv python install 3.13`)
- **uv** package manager (build backend is `uv_build`)
- Virtual environment at `.venv/` (created with `uv venv --python 3.13 .venv`)

### Development Commands

- **Activate venv**: `source .venv/bin/activate`
- **Install (editable)**: `uv pip install -e .`
- **Run tests**: `python -m unittest discover test -v`
- **Run examples**: `MPLBACKEND=Agg python examples/twocolor.py` (see Known Issues below)

### Known Issues

- The example scripts in `examples/` (e.g., `twocolor.py`, `tirf.py`) call `numpy.in1d`, which was removed in NumPy 2.0+. These examples will fail at runtime. The unit tests in `test/` do not exercise this code path and pass successfully.
- Set `MPLBACKEND=Agg` when running scripts that use matplotlib in a headless environment.

### No Lint Configuration

This repository does not include a linter configuration (no `ruff`, `flake8`, `pylint`, `mypy`, or similar). Linting is not part of the CI pipeline.
