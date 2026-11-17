# Project-local environments (D168 Step A0, rebuilt 2026-11-17)

Both environments are git-ignored and were lost when the session container was replaced on 2026-11-15/17. They were
rebuilt on 2026-11-17 on the owner's approval (D168, step A0). These files make the rebuild exact.

## Python: `.venv/`

- Python 3.13.18; `python -m venv .venv`, then
  `pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu` and `pip install -e . --no-deps`.
- `pip freeze` was compared with `requirements-lock.txt` after the rebuild: identical.
- PySR installs its Julia backend into `.venv/julia_env/` on first import (Julia 1.13.4, SymbolicRegression 2.7.2).

## R: `.rlocal/`

- micromamba 2.10.0 (`.rlocal/bin/micromamba`), `MAMBA_ROOT_PREFIX=.rlocal/mamba`.
- conda-forge environment `.rlocal/env`, exact package list in **`env/rlocal_conda_explicit_lock.txt`**
  (`micromamba create -p .rlocal/env --file env/rlocal_conda_explicit_lock.txt` recreates it). Requested specs:
  `r-base=4.4.3 r-survival r-nlme r-jsonlite r-matrix compilers make` (channel conda-forge only).
- **`mgcv` 1.9-5**, installed from the CRAN source tarball `mgcv_1.9-5.tar.gz`
  (MD5 `5a88e8ede6eab594a0399a8329f580e9`, equal to the CRAN `PACKAGES` index on 2026-11-17;
  SHA-256 `d079c68e500708de4df79692e83493636004915dc6689cd7350e226291abd992`) with
  `R CMD INSTALL`. conda-forge offers only up to `r-mgcv` 1.9-4 for R 4.4; 1.9-5 was chosen because its changelog
  modifies `ti` terms ("ti terms modified to never trigger numerical side constraints") and tensor-product penalty
  scaling, i.e. exactly the construction used by the AFCM reference, and because 1.9-5 is the version whose source was
  read for the D168 protocol.
- Installed R packages and versions: **`env/r_installed_packages.tsv`**. Key versions: R 4.4.3 (2025-02-28),
  mgcv 1.9-5, survival 3.8-12, nlme 3.1-170, jsonlite 2.0.0, Matrix 1.7-6, splines 4.4.3.
- **Not installed** (held back by the owner until Stage C is reviewed): JMbayes2 and its dependencies (Round 24 /
  Study B).
