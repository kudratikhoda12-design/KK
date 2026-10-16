# Execution log (resumable)

Human-readable log of the autonomous staged run (D21). Machine-readable command records are in
`reports/logs/commands.jsonl`; full outputs are in `reports/logs/*.log`.

## 2026-10-16 — Session start
- Repository inspected: branch `claude/blissful-faraday-2unozw`, clean tree, HEAD `f54df73` (Initial commit;
  only `README.md`). README left unchanged (D28).
- Resources: 4 CPUs, 15 GB RAM, ~29 GB free disk, no GPU. Tools: pandoc, pdftotext, pdftoppm, Chromium.
- Global `pip list` snapshot saved before any install (89 lines) for criterion E-1.
- Draft documents copied into `docs/` and revised (revision 2): D14-R..D28 recorded in `docs/decisions.md`.

## Stage A — Environment and survival likelihood: PASSED (attempt 3)
- `scripts/setup_env.sh` built `.venv` (exit 0, 116.6 s): torch 2.14.1+cpu, pandas 2.3.3, lifelines 0.30.3,
  scikit-survival 0.28.0, rdata 1.1.0 (added for Stage G; recorded in D29). `pip check`: no broken requirements.
- `scripts/verify_env.sh` (exit 0): E-1 global pip list unchanged (empty diff); E-5 throwaway venv from the lock
  file reproduces `pip freeze` exactly.
- Tests attempt 1: 41 passed / 4 failed.
  - A2 (x3): NaN from `quad` over (-inf, inf) — `exp(u)` overflow in the test integrand (harness bug).
  - A7 log-logistic: our max log-lik exceeded lifelines' by 7.8e-4 (reference optimiser not converged).
- Repair 1: A2 finite range [-700, 700]; A7 `gtol` for lifelines + diagnostic (our likelihood at lifelines'
  estimates reproduces lifelines' value within 1e-8).
- Attempt 2: 41 passed / 4 failed — A2 quadrature missed narrow peak (sigma=0.2); A7 SLSQP ignores `gtol`,
  BFGS (log-normal) raised ConvergenceError with gtol=1e-10.
- Repair 2: A2 integrate in w = (u - eta)/sigma over [-60, 60] with breakpoints; A7 `ftol=1e-12` for SLSQP only.
- Attempt 3: **45 passed, 0 failed** (log `reports/logs/stageA_tests_attempt3.log`). No threshold changed.
  The likelihood implementation (`src/kanfae/survival/aft.py`) was not modified by either repair.

## Stage B — Synthetic data: PASSED (attempt 1)
- `src/kanfae/data/schema.py` (Observed/Oracle, `to_training_arrays` rejects Oracle), `src/kanfae/data/synthetic.py`.
- `tests/test_synthetic.py`: **40 passed, 0 failed** (log `reports/logs/stageB_tests_attempt1.log`).
  STAT outcomes at pre-specified seeds: G3 realised censoring 0.0000 / 0.3027 / 0.5935 (targets 0/0.3/0.6);
  G8 KS p = 0.583 (Weibull), 0.788 (log-normal), 0.583 (log-logistic); G9 corr = -0.0034; G10 max dev 0.00175;
  G12 masked fraction 0.2002.
- Note: identical KS p-values for Weibull and log-logistic are expected: numpy draws both by inverse CDF from the
  same uniform stream (common random numbers) and the KS statistic is invariant under U -> 1-U reflection.
