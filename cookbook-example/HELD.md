# Held: the LuxorLabs/cookbook example for this provider

This folder is `examples/nemo-gym-tenki` for LuxorLabs/cookbook, parked here
because it installs this provider from git and this repo is private, so the
cookbook's CI and readers can't install it yet. It passes live against NeMo
Gym 0.7.0 from PyPI (verify.py and run.py, 2026-10-16).

To land it once this repo is public (or the package is on PyPI):
1. Copy this folder to `examples/nemo-gym-tenki/` in the cookbook (drop this file).
2. Bump the provider pin in pyproject.toml to the latest commit, run `uv lock`.
3. Add its row to the cookbook README's "Agent platforms" table.
