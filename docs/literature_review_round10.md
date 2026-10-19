# Literature review, Round 10 (2026-10-19): functional and longitudinal survival models, KAN survival models

**Scope and method.** This is a targeted review, not a systematic one: no database protocol, no screening log, no
double screening. It extends the round-2 review (`docs/literature_review.md`, kept unchanged). The questions are those
the owner set in D110:
- What exists for survival analysis with functional or longitudinal covariates (functional linear Cox, additive
  functional Cox, joint models, landmarking, deep dynamic models)?
- What exists for KAN-based survival models?
- What exists for KAN-based functional regression?
- Where do the planned Functional CoxKAN / KAN-AFT models (B1, B2, B3) sit relative to all of these?

Each item was verified by a web search, its landing page, its arXiv abstract page, or Crossref metadata
(api.crossref.org), as marked in the tables. "Abstract" means only the abstract or metadata was read, so claims about
methodological detail beyond it are not made. Search date: 2026-10-19.

**Search strings used.**
1. "Additive functional Cox model" Cui Crainiceanu Leroux
2. Cox regression models with functional covariates (Gellar et al.)
3. FLCRM functional linear Cox regression model
4. Dynamic-DeepHit longitudinal competing risks
5. Kolmogorov-Arnold network functional data / longitudinal covariates survival
6. functional Kolmogorov-Arnold network functional regression
7. van Houwelingen landmarking; landmark supermodel
8. JM R package joint modelling aids prothro
9. Yao Müller Wang PACE sparse longitudinal
10. Functional generalized additive models (McLean et al.)
11. CoxKAN; KAN-AFT; SurvKAN; KAPLAN
12. neural additive models for survival
13. deep learning with functional covariates for survival
14. Crossref look-ups for Wulfsohn & Tsiatis, Putter & van Houwelingen, Katzman et al., van Houwelingen (2007) and
    F-KAN

## 1. Functional-covariate survival models (non-KAN)

| Reference | What it does | Relevance to this project | Verified |
|---|---|---|---|
| Gellar, Colantuoni, Needham & Crainiceanu (2015), *Cox regression models with functional covariates for survival data*, Statistical Modelling 15(3):256–278, doi:10.1177/1471082X14565526 | Linear functional Cox model: log-hazard contains ∫β(t)X(t)dt for a densely sampled baseline functional covariate. Penalised signal regression, penalised partial likelihood; R package `pcox` | The **linear** functional Cox baseline. Our `fpca_cox` is the FPCA-basis version of the same model class | Journal abstract page, PMC record |
| Kong, Ibrahim, Lee & Zhu (2018), *FLCRM: Functional linear Cox regression model*, Biometrics 74(1):109–117, doi:10.1111/biom.12748 | FPCA of the functional predictor plus a Cox regression on the scores and on scalar covariates; convergence rates; score test for a zero slope function | **B1's linear analogue** (FPCA scores into Cox). B1 replaces the linear Cox on the scores with a KAN | Abstract / RePEc record |
| Cui, Crainiceanu & Leroux (2021), *Additive Functional Cox Model*, J. Computational and Graphical Statistics 30(3):780–793, doi:10.1080/10618600.2020.1853550 | Additive functional Cox model (AFCM): log-hazard contains ∫F(X(s), s)ds, with F a bivariate smooth estimated by penalised tensor-product splines. Uses transformations of X (e.g. quantile) to address weak identifiability where data are sparse | **The direct non-KAN competitor of B2.** A single-layer B2 under the Cox loss has the same model form (∫φ(X(t), t)dt); only the parametrisation, training and regularisation differ | PMC author manuscript (PMC8664082), Crossref |
| McLean, Hooker, Staicu, Scheipl & Ruppert (2014), *Functional Generalized Additive Models*, JCGS 23(1):249–269, doi:10.1080/10618600.2012.729985 | Scalar-on-function regression with ∫F(X(t), t)dt (FGAM); tensor-product B-splines with roughness penalties; quantile transformation; R package `refund` | The origin of the ∫F(X(t), t)dt form that AFCM brings to survival | PMC record (PMC3982924) |
| Du & Leroux (2024), *Time-Varying Functional Cox Model*, arXiv:2412.14478 | Linear functional Cox with a coefficient varying in the functional domain *and* event time; Cox–Poisson and landmark estimation via `mgcv` | Time-varying functional effects; a possible extension, not a planned baseline | arXiv abstract |
| Yao, Müller & Wang (2005), *Functional Data Analysis for Sparse Longitudinal Data*, JASA 100:577–590, doi:10.1198/016214504000001745 | PACE: pooled mean and covariance smoothing, FPC scores by conditional expectation; works with few irregular measurements per subject (applied to CD4 data) | **The sparse-data front end for B1 and B2** (S-Sparse, aids, prothro) | Abstract, author-hosted PDF listing |

## 2. Joint models, landmarking and dynamic prediction

| Reference | What it does | Relevance | Verified |
|---|---|---|---|
| Wulfsohn & Tsiatis (1997), *A Joint Model for Survival and Longitudinal Data Measured with Error*, Biometrics 53(1):330–, doi:10.2307/2533118 | Shared random-effects joint model: Cox survival with a longitudinal covariate measured with error; EM estimation | Classical joint model | Crossref |
| Tsiatis & Davidian (2004), *Joint modeling of longitudinal and time-to-event data: an overview*, Statistica Sinica 14:809–834 | Review of joint models with shared latent random effects | Background | Duke record |
| Rizopoulos (2010), *JM: An R Package for the Joint Modelling of Longitudinal and Time-to-Event Data*, J. Statistical Software 35(9):1–33, doi:10.18637/jss.v035.i09 | Shared-parameter joint models; bundles the `aids`, `prothro` and `pbc2` data | Source of the `aids` and `prothro` data sets (Round 13); joint-model reference. **R is not installed here**, so the round-13 joint-model-style baseline is a two-stage mixed-effects re-implementation, not JM itself (D112) | JSS page; CRAN listing (JM 1.5-2, 2022-08-08); package manual pages for `aids`, `prothro` |
| van Houwelingen (2007), *Dynamic Prediction by Landmarking in Event History Analysis*, Scandinavian J. Statistics 34(1):70–85, doi:10.1111/j.1467-9469.2006.00529.x | Landmarking: fit a survival model at a landmark time to subjects still at risk, using covariate values up to the landmark | **B3** and the landmark Cox baseline | Crossref |
| van Houwelingen & Putter (2012), *Dynamic Prediction in Clinical Survival Analysis*, Chapman & Hall/CRC (some sources give 2011) | Book; includes the **landmark supermodel**: one model fitted to the stacked landmark data, with landmark-dependent effects and baseline | **B3's stacked supermodel design** | Library and publisher records; the slide deck attributing the supermodel to the book |
| Putter & van Houwelingen (2022), *Landmarking 2.0: Bridging the gap between joint models and landmarking*, Statistics in Medicine 41(11):1901–1917, doi:10.1002/sim.9336 | Landmarking that models the longitudinal process at the landmark; comparison with joint models | Design reference for B3 | Crossref |
| Lee, Yoon & van der Schaar (2020), *Dynamic-DeepHit*, IEEE Trans. Biomedical Engineering 67(1):122–133, doi:10.1109/TBME.2019.2909027 | Deep dynamic survival with competing risks from the full history of repeated measurements; no assumed longitudinal model | Deep dynamic-prediction reference (not implemented as a baseline; disclosed as a limitation) | Journal listing, search summaries |
| Katzman et al. (2018), *DeepSurv*, BMC Medical Research Methodology 18, doi:10.1186/s12874-018-0482-1 | MLP Cox model on static covariates | Background for neural Cox | Crossref |
| Hu & Nan (2025), *Nonparametric Estimation of Conditional Survival Function with Time-Varying Covariates Using DeepONet*, arXiv:2505.22748 | Hazard as a nonlinear operator of entire covariate histories (DeepONet); full-likelihood loss; ADNI application | Closest deep "operator on X(·)" model; not interpretable by design | arXiv abstract |
| Gao et al. (2025), *Deep Survival Analysis for Competing Risk Modeling with Functional Covariates and Missing Data Imputation* (FCRN), arXiv:2509.25381 | Discrete-time competing-risks deep model with a basis layer for functional covariates | Deep functional survival; interpretability not discussed | arXiv abstract |

## 3. KAN and KAN-based survival models

| Reference | What it does | Inputs | Verified |
|---|---|---|---|
| Liu et al. (2024/2025), *KAN: Kolmogorov–Arnold Networks*, arXiv:2404.19756, ICLR 2025 | Learnable univariate spline functions on edges; pruning; symbolic regression | – | arXiv, ICLR page |
| Knottenbelt et al. (2025), *CoxKAN*, Bioinformatics 41(8), btaf413; arXiv:2409.04290 | KAN for the Cox log-partial hazard; L1 + entropy + coefficient regularisation; pruning by L1 threshold (validation-chosen); auto-symbolic (linear R² > 0.99, then a 22-function library), PySR optional; uses a *fast approximation* of the Cox loss that is slightly inaccurate under heavy ties; 13 data sets | **Static** covariates only. The authors list inheriting the PH assumption as a limitation | arXiv full text (HTML v1, first 100,000 characters); round-2 full text (v2) |
| Jose, Francis & Kattumannil (2025/2026), *KAN-AFT* (arXiv:2512.20305; v2 title "A Structured Nonparametric Framework for Nonlinear Accelerated Failure Time Models") | KAN for the AFT location; Buckley–James, IPCW and transformation estimators; pruning; symbolic forms | **Static** covariates | arXiv abstract; round-2 full text |
| Mastroleo et al. (2026), *SurvKAN*, arXiv:2602.02179 | Fully parametric, time-continuous survival KAN; time as an input; log-hazard output | Covariate type not stated; time-varying *effects*, not time-varying covariates | arXiv abstract |
| Boulitsakis Logothetis, Wood & Liò (2026), *KAPLAN*, arXiv:2605.23082 | B-spline KAN for the conditional hazard as a joint function of covariates and time; a single layer reduces to a GAM | Time-varying *effects*; repeated measurements not addressed | arXiv abstract |
| Lundin, Chen, Schulz & Tao (2025), *AI-based dynamic predictions of longitudinal and survival data of Alzheimer's disease*, Alzheimer's & Dementia 21(S2):e103788, doi:10.1002/alz70856_103788 | **JM-KAN**: a KAN-based joint model of longitudinal and survival data; yearly landmarks; NACC data; integrated AUC 0.789, IBS 0.118 | **Longitudinal** covariates with landmarking. How the trajectories enter the KAN, the loss and any interpretability analysis are **not stated in the available text** (conference abstract) | PMC page (PMC12737850) |
| Neural additive models for survival (SurvNAM, TimeNAM, PseudoNAM, DNAMite) | Per-feature networks giving shape functions under Cox/GAM-type survival losses | Static covariates; the NAM analogue of a single-layer KAN | Search summaries only |

## 4. KAN-based functional regression (non-survival)

| Reference | What it does | Verified |
|---|---|---|
| Qin & Li (2026), *F-KAN: a novel framework for functional regression based on Kolmogorov–Arnold networks*, Computational Statistics 41(5), article 114, doi:10.1007/s00180-026-01791-x | Per search summaries: a nonlinear extension of functional principal component regression, i.e. FPC scores into a KAN; consistency of the estimator; spectroscopy applications; conformal prediction intervals. **Scalar, not survival, response** | Crossref (title, authors, venue); method summary from search results only, full text not accessible |
| Cao M. (supervisor Cao J.), FunKAN (SFU thesis defence) | KAN for function-to-function learning with reconstruction and regularisation | Defence page only (round-2 review) |
| FunKAN, arXiv:2509.13508 | Functional KAN for medical image enhancement and segmentation | Search abstract |

## 5. Gap analysis (what the planned models add, and what they do not)

| Planned model | Closest existing work | Assessment |
|---|---|---|
| **Real CoxKAN** (exact Breslow/Efron partial likelihood, Breslow baseline) | CoxKAN (Knottenbelt et al.) | **Not new.** A re-implementation with exact tie handling instead of CoxKAN's fast approximation, which is a documented deviation |
| **B1**: FPCA / PACE scores, or trajectory summaries, into CoxKAN or KAN-AFT | FLCRM (FPCA → linear Cox); F-KAN (FPCA → KAN, scalar response); CoxKAN/KAN-AFT (KAN on static covariates) | **A direct composition of published components.** We found no paper applying FPCA scores → KAN under a censored-survival loss, but this is an incremental combination, not a methodological novelty |
| **B2**: η = Σ_d ∫ φ_d(X_d(t), t) dt, with φ_d a KAN, under the Cox and AFT losses; homogeneous special case ∫φ_d(X_d(t))dt | AFCM (Cui et al., Cox, tensor-product penalised splines); FGAM (non-survival) | **Single-layer B2 under the Cox loss has the same model form as AFCM.** Its differences are the parametrisation (KAN edges, separable or KAN-composed φ), training (gradient-based with L1/entropy regularisation instead of REML), pruning, symbolic extraction, and the option to compose layers (B2-deep). An **AFT version** of the additive functional model was not found in this search, which is not proof that it does not exist. Any claim must be about the empirical comparison with AFCM, not about the model form |
| **B3**: stacked landmark supermodel with KAN effects (landmark time as an input) | Landmark supermodel (van Houwelingen & Putter); JM-KAN (KAN + landmarks; method details unavailable) | **Partly covered by JM-KAN**, whose design cannot be compared from the available abstract. No novelty claim |
| **KAN-FAE** (Rounds 1–9) | Functional autoencoders (Wu et al. 2024); FunKAN; autoencoder-survival models | Already reviewed in round 2. Round 9 found no advantage over the MLP; it serves as the paper's motivating negative result |

**Claim policy (D111).** No novelty claim will be made from this targeted review. The paper's contributions will be
stated as:
- an empirical, pre-registered comparison of functional KAN survival models against established functional survival
  methods, in particular AFCM;
- the shape-recovery, selection and symbolic-interpretability evaluation;
- the negative architectural result for KAN-FAE;
- a fully reproducible benchmark.

Any wording such as "first" requires a systematic review first. That review should include the full text of JM-KAN,
F-KAN, FunKAN and KAPLAN.

## 6. Data sources identified (to be acquired in Round 13; not downloaded in Round 10)

| Data set | Source | Content (per package documentation) | Notes |
|---|---|---|---|
| `pbcseq` | R package `survival` (already acquired in round 1: `data/raw/pbc`) | Mayo Clinic PBC, repeated laboratory measurements | Used in rounds 1–9 (single 730-day landmark) |
| `prothro` / `prothros` | R package JM 1.5-2 (CRAN tarball reachable, HTTP 206 on a range request, 2026-10-19) | Prednisone vs placebo in liver cirrhosis; longitudinal prothrombin index; time to death; treatment | The documentation is inconsistent: 488 patients in the description, 467 in the format section. To be checked on acquisition. Original reference: Andersen, Borgan, Gill & Keiding (1993) |
| `aids` / `aids.id` | R package JM 1.5-2 | ddI vs ddC trial; 467 patients; 1,408 CD4 observations; time to death; drug, gender, prevOI, AZT | References: Goldman et al. (1996), JAIDS 11:161–169; Guo & Carlin (2004), Am. Statistician 58:16–24 |
| C-MAPSS FD001 | NASA (already acquired in round 1) | Turbofan run-to-failure | Secondary engineering data set |
