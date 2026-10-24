**symbolic_distill** (B2-S Cox, seed-0 member after pruning; identifiable RISE; validation NLPL is the unpenalised loss)

| Data | Model | Mean RISE | RISE φ₁ | RISE φ₂ | Curve error | IBS | Validation NLPL |
|---|---|---|---|---|---|---|---|
| S-Add 961 | after pruning | 0.197 | 0.292 | 0.103 | 0.0531 | 0.1923 | 4.2205 |
| S-Add 961 | sub_noguard | 0.197 | 0.292 | 0.103 | 0.0518 | 0.1924 | 4.2205 |
| S-Add 961 | refit_old | 1.781 | 2.360 | 1.201 | 0.0906 | 0.2081 | 4.2925 |
| S-Add 961 | sub_guard | 0.197 | 0.292 | 0.103 | 0.0518 | 0.1924 | 4.2205 |
| S-Add 961 | refit_guard | 1.781 | 2.360 | 1.201 | 0.0906 | 0.2081 | 4.2925 |
| S-Add 961 | distill_mu0.01 | 0.593 | 0.573 | 0.613 | 0.0717 | 0.1996 | 4.2463 |
| S-Add 961 | distill_mu0.1 | 0.239 | 0.310 | 0.168 | 0.0573 | 0.1934 | 4.2179 |
| S-Add 961 | distill_mu1 | 0.200 | 0.293 | 0.107 | 0.0524 | 0.1924 | 4.2151 |
| S-Add 961 | distill | 0.200 | 0.293 | 0.107 | 0.0524 | 0.1924 | 4.2151 |
| S-Sparse 962 | after pruning | 0.503 | 0.828 | 0.178 | 0.0862 | 0.1757 | 4.2477 |
| S-Sparse 962 | sub_noguard | 0.503 | 0.830 | 0.176 | 0.0861 | 0.1750 | 4.2477 |
| S-Sparse 962 | refit_old | 5.184 | 6.514 | 3.854 | 0.1126 | 0.1893 | 4.3814 |
| S-Sparse 962 | sub_guard | 0.503 | 0.830 | 0.176 | 0.0861 | 0.1750 | 4.2477 |
| S-Sparse 962 | refit_guard | 5.184 | 6.514 | 3.854 | 0.1126 | 0.1893 | 4.3814 |
| S-Sparse 962 | distill_mu0.01 | 0.675 | 1.007 | 0.343 | 0.0926 | 0.1805 | 4.2875 |
| S-Sparse 962 | distill_mu0.1 | 0.507 | 0.843 | 0.172 | 0.0868 | 0.1770 | 4.2520 |
| S-Sparse 962 | distill_mu1 | 0.502 | 0.828 | 0.177 | 0.0860 | 0.1755 | 4.2479 |
| S-Sparse 962 | distill | 0.502 | 0.828 | 0.177 | 0.0860 | 0.1755 | 4.2479 |

**b2s_res_cox and the anchored follow-up** (curve error only on simulated data)

| Data | Model | Curve error | IBS | Harrell C | Uno C | D-cal p | Functional channels kept by the gate |
|---|---|---|---|---|---|---|---|
| S-Last 963 | b2s_anchor_cox | 0.1736 | 0.1708 | 0.6377 | 0.6438 | 0.746 | [0, 0, 0] |
| S-Sparse 964 | b2s_anchor_cox | 0.0810 | 0.1679 | 0.6614 | 0.6652 | 0.426 | [1, 1, 1] |
| dev_aids 965 | b2s_anchor_cox | – | 0.1993 | 0.5950 | 0.5928 | 0.790 | [0, 0, 0] |
| S-Last 963 | b2s_cox | 0.1800 | 0.1731 | 0.6252 | 0.6359 | 0.263 |  |
| S-Last 963 | static_coxkan | 0.1679 | 0.1699 | 0.6394 | 0.6455 | 0.785 |  |
| S-Last 963 | landmark_cox | 0.1807 | 0.1728 | 0.6357 | 0.6395 | 0.719 |  |
| S-Last 963 | b2s_res_cox | 0.1727 | 0.1702 | 0.6403 | 0.6462 | 0.733 | [0, 0, 0] |
| S-Sparse 964 | b2s_cox | 0.0828 | 0.1684 | 0.6614 | 0.6645 | 0.128 |  |
| S-Sparse 964 | static_coxkan | 0.1131 | 0.1862 | 0.5773 | 0.5792 | 0.995 |  |
| S-Sparse 964 | landmark_cox | 0.1161 | 0.1895 | 0.5601 | 0.5641 | 0.918 |  |
| S-Sparse 964 | b2s_res_cox | 0.0924 | 0.1722 | 0.6352 | 0.6366 | 0.775 | [1, 1, 0] |
| dev_aids 965 | b2s_cox | – | 0.1946 | 0.6266 | 0.6353 | 0.638 |  |
| dev_aids 965 | static_coxkan | – | 0.1987 | 0.6022 | 0.5953 | 0.668 |  |
| dev_aids 965 | landmark_cox | – | 0.2055 | 0.5826 | 0.5883 | 0.375 |  |
| dev_aids 965 | b2s_res_cox | – | 0.1985 | 0.6055 | 0.5997 | 0.686 | [0, 0, 0] |

**afcm_ens and alternatives** (the CK-6c procedure, n 2,000; identifiable RISE; CK-6c criterion φ₁ < 0.15 and φ₂ < 0.10)

| Seed | AFCM variant | RISE φ₁ | RISE φ₂ | Selected λ_t |
|---|---|---|---|---|
| 22 | V3 (single fit) | 0.247 | 0.053 | 0.01 |
| 22 | V3, D120 parsimonious λ rule (tol 0.01) | 0.247 | 0.053 | 0.01 |
| 22 | V3, per-channel λ_t | 0.240 | 0.108 | [0.01, 0.0001, 1.0] |
| 22 | afcm_ens: 3-resample bag of V3 | 0.239 | 0.050 | [0.01, 0.01, 0.01] |
| 914 | V3 (single fit) | 0.023 | 0.053 | 1.0 |
| 914 | V3, D120 parsimonious λ rule (tol 0.01) | 0.092 | 0.056 | 10000.0 |
| 914 | V3, per-channel λ_t | 0.224 | 0.048 | [0.0001, 1.0, 0.01] |
| 914 | afcm_ens: 3-resample bag of V3 | 0.056 | 0.064 | [10000.0, 100.0, 10000.0] |
| 966 | V3 (single fit) | 0.055 | 0.056 | 10000.0 |
| 966 | V3, D120 parsimonious λ rule (tol 0.01) | 0.063 | 0.056 | 10000.0 |
| 966 | V3, per-channel λ_t | 0.065 | 0.056 | [10000.0, 10000.0, 0.0001] |
| 966 | afcm_ens: 3-resample bag of V3 | 0.097 | 0.055 | [10000.0, 0.01, 10000.0] |
