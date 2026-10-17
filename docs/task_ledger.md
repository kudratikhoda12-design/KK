# Task ledger: originally failed, blocked, skipped or unrun items (opened 2026-10-17)

Each row is updated as work proceeds. The final column points to the supporting log or record. Original
statuses come from `reports/FINAL_STATUS.md` (commit `a918a56`) and the gate re-run of revision 3.

| # | Item | Original status / reason | Blocking dependency | Alternatives considered | Experiments / tests | Final outcome | Evidence | Remaining limitation |
|---|---|---|---|---|---|---|---|---|
| L1 | Learnability gate R2 (functional MLP AE) | FAILED (original design, 2 repairs: C 0.8361); FAILED again with revision 3 NFL front (C 0.8461) | – | (see L1 search) | pending | pending | `reports/logs/stageD_gate_attempt*.log`, `rev3_gate_attempt1.log` | |
| L2 | H1 primary: KAN vs MLP functional AE | NOT RUN (blocked by L1, D33) | L1 | | pending | pending | | |
| L3 | H2: α > 0 vs α = 0 | NOT RUN (blocked by L1) | L1 | | pending | pending | | |
| L4 | H3: functional vs flattened under sparse sampling | NOT RUN (blocked by L1) | L1 | | pending | pending | | |
| L5 | Training controls S1/S2/S3/survival-only on the revised design | NOT RUN (blocked by L1) | L1 | | pending | pending | | |
| L6 | Ablations AB1–AB8 | NOT RUN (blocked by L1) | L1 | | pending | pending | | |
| L7 | Neural models on C-MAPSS and PBC | NOT RUN (blocked by L1) | L1, L2 | | pending | pending | | |
| L8 | KAN regularisation and pruning | NOT IMPLEMENTED (audit finding A-KAN-REG) | – | | pending | pending | | |
| L9 | Symbolic formula extraction (R² > 0.99) | NOT IMPLEMENTED (audit finding A-SYM) | L8 | | pending | pending | | |
| L10 | Censoring-adjusted least-squares estimators (BJ, IPCW, transform) | NOT IMPLEMENTED (new requirement) | – | | pending | pending | | |
| L11 | M2 for the linear autoencoder | SKIPPED: rank-limited, not applicable (D32) | – | none needed; a mathematical impossibility, not a defect | – | remains NOT APPLICABLE | `reports/logs/stageCDE_unit_final.log` | |
| L12 | PBC endpoint choice (E1 / E2 / competing risks) | UNRESOLVED owner decision (D9-R) | owner | | – | pending | | |
| L13 | C-MAPSS landmark design (A / B) | UNRESOLVED owner decision (D3b-R) | owner | | – | pending | | |
| L14 | Interval likelihood for integer cycles | NOT IMPLEMENTED (D25) | – | | pending | pending | | |
| L15 | C-MAPSS FD002–FD004 | NOT RUN (budget) | – | | pending | pending | | |
| L16 | Informative-censoring and misspecification synthetic variants | NOT RUN | – | | pending | pending | | |
| L17 | Literature review / bibliographic verification | NOT DONE (D27) | – | | pending | pending | | |
| L18 | Diagnostic run `diag_rev3_R2` (NFL variants) | STOPPED by me after 1 of 5 configurations, to free CPU; superseded by the controlled search in L1 | – | – | – | STOPPED (partial log kept) | `reports/logs/diag_rev3_R2.log` | |
