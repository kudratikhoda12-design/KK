# Literature review (targeted; 2026-10-17)

**Scope and method.** This is a targeted search, not a systematic review. It used the four web searches logged in
the session, verification of key items on their landing pages, and full-text reading of the three downloaded
PDFs (SHA-256 checksums in `reports/run_manifest.json`):
- KAN-AFT, arXiv:2512.20305v2;
- CoxKAN, arXiv:2409.04290v2 (journal version: Bioinformatics 41(8), btaf413, 2025);
- Wu et al., arXiv:2401.09499.

**The uploaded PDFs (`project TS_2(1).pdf`, `Project-TS_1(3).pdf`) were not present in this session's file
system.** The public arXiv versions were used instead. They may differ from the uploaded copies; for example,
the CoxKAN journal version may differ from arXiv v2.

## Directly relevant prior work

| Work | What it does | Relation to KAN-FAE | Verified how |
|---|---|---|---|
| Wu, Beaulac & Cao (2024), *Functional Autoencoder for Smoothing and Representation Learning*, arXiv:2401.09499 | A functional autoencoder. The encoder projection layer takes weighted inner products of the curves with learned functional weights; the decoder maps back through fixed basis functions; handles irregular sampling. | **Our original front end (A0) and decoder follow this design.** Our functional linear layer `∫β(t)x(t)dt` is the same idea. | arXiv landing page and PDF |
| Cao M. (supervisor Cao J.), *FunKAN: Interpretable Kolmogorov–Arnold Networks for Learning* (SFU thesis defence; year not stated on page) | A KAN-based framework for nonlinear functional learning (functional input to functional output). Stages: functional representation, nonlinear mapping, function reconstruction and structural regularisation, plus spline refinement and attribution-guided pruning. | **Strong conceptual overlap** with "KAN + functional representation + reconstruction + regularisation". No survival component is mentioned. | SFU defence announcement page only; no full text found |
| FunKAN, arXiv:2509.13508 | A functional generalisation of the Kolmogorov–Arnold representation for medical image enhancement and segmentation (Hermite-basis inner functions). | Same name, different domain (imaging). | Search result abstract |
| Knottenbelt et al. (2025), CoxKAN, Bioinformatics 41(8) btaf413; arXiv:2409.04290 | A KAN for the Cox log-partial hazard. Regularisation (L1 / entropy / coefficient), L1 pruning, progressive symbolic fitting (linear R² > 0.99, then a library, then PySR). | **Adopted:** the regularisation definitions, pruning by L1 threshold and progressive symbolic fitting with the R² > 0.99 acceptance. **Not adopted:** the Cox partial likelihood (our model is AFT). | Full text (arXiv v2) |
| Jose, Francis & Kattumannil (2026), KAN-AFT, arXiv:2512.20305v2 | A KAN for the nonlinear AFT location `η(z)` under right censoring via Buckley–James, IPCW or transform least squares. L_R regularisation, edge/node pruning (θ = 1e-2), affine symbolic fitting, then PySR. | **Adopted:** the three censoring-adjusted estimators as separate variants (with documented deviations, D42), node pruning, staged symbolic fitting and the simulation-style validation. | Full text (v2) |
| Mastroleo et al. (2026), SurvKAN, arXiv:2602.02179 | A fully parametric, time-continuous survival KAN (time as an input; log-hazard output). | KAN survival model on scalar covariates; not functional. | arXiv landing page |
| KAPLAN-HR, arXiv:2605.23082 | A B-spline KAN for the conditional hazard as a function of covariates and time. | KAN survival on scalar covariates. | Search result abstract only |
| Shen et al. (2024), AESurv, Briefings in Bioinformatics 25(6) bbae479 | An autoencoder learns low-dimensional representations of high-dimensional DNA methylation plus clinical factors for coronary heart disease time-to-event prediction. | Autoencoder plus survival, with **vector** (not functional) inputs. | Search results (journal page not fetched) |
| Apellániz et al. (2024), SAVAE, Scientific Reports 14 (doi:10.1038/s41598-024-76047-z) | A VAE with a survival-tailored objective; parametric event-time distributions with differentiable likelihoods. | Generative autoencoder plus parametric survival likelihood, with vector inputs. Closest in spirit to our joint reconstruction plus parametric AFT likelihood. | Search results |
| Lin & Luo (2022), Statistics in Medicine (review); TransformerJM | Machine learning for multivariate longitudinal data in time-to-event prediction, including multivariate FPCA scores fed to DeepSurv. | Our FPCA + Cox/AFT baselines follow this conventional pipeline. | Search result abstract |
| Functional Competing Risk Net (arXiv 2509.25381); DeepONet hazard model (arXiv 2505.22748); Dynamic-DeepHit (Lee et al. 2019) | Deep survival models built on functional or longitudinal covariates. | Alternative architectures for the same problem class. | Search result abstracts |

## Consequence for claims

- Functional autoencoders (Wu et al.), KAN survival models (CoxKAN, KAN-AFT, SurvKAN, KAPLAN-HR),
  autoencoder-survival models (AESurv, SAVAE) and KAN-based functional learning with reconstruction (FunKAN)
  all exist.
- Combining a functional KAN autoencoder with a censoring-aware AFT likelihood was not found as a single
  published method in this targeted search. However, the components are all established, and FunKAN's full text
  could not be checked.
- **No novelty claim is made.** A claim would need a systematic review (databases, search strings, screening
  log), which this is not.

## Known reference discrepancies (recorded, not resolved)

- KAN-AFT cites "Fan & Gijbels [8]" for the transformation estimator. Its reference [8] is Fan, Gijbels & King
  (1997), *Local likelihood and local partial likelihood in hazard regression*, which does not appear to be the
  synthetic-data transformation source.
- KAN-AFT states CA-1 (`C ⊥ T | z`) but uses a marginal Kaplan–Meier estimate of censoring in IPCW and the
  transform. That is consistent only under CA-2.
- KAN-AFT's Buckley–James imputes on the time scale and then takes logs, unlike the classical log-scale
  imputation.
- The CoxKAN 22-function symbolic library is defined only in its supplementary material, which was not obtained.
  Our library is therefore our own documented list (`src/kanfae/symbolic/fit.py`).
