Contributing to Raptor is easy: just open a pull request. Make `main` the
destination branch on the Raptor repository and allow edits from maintainers.

Your pull request must work with all current Raptor tutorial examples and be
reviewed by at least one of the main developers.

Create an editable development environment and run the same checks as CI:

```bash
python -m pip install -e ".[test,dev]"
pre-commit run --all-files
NUMBA_NUM_THREADS=2 python -m pytest -q
python -m build
```

The Python source limit is 80 columns. Black formatting, spelling, merge
conflict, whitespace, and line-length checks are enforced by `pre-commit`.
