# Contributing to LM-Fix

Thanks for your interest in contributing!

## Getting started
1. Fork the repo and create a new branch: `git checkout -b feat/your-feature`.
2. Install dev deps: `pip install -r requirements.txt` (and optionally `pip install -r requirements-dev.txt`).
3. Run linters/tests (if present): `ruff .` and `pytest`.

## Coding style
- Follow PEP 8/PEP 257.
- Prefer `logging` over `print`.
- Add type hints for new/changed functions.

## Pull requests
- Keep PRs focused and small.
- Include a clear description, screenshots for UI/plots, and benchmarks when appropriate.
- Link related issues.

## Security
Please **do not** file security bugs in public issues. See `SECURITY.md` for responsible disclosure.
