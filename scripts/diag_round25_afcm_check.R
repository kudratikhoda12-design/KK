# Round 25 readiness audit, Task 1D: independent R check of the Python AFCM's mechanics on identical saved inputs
# (written by scripts/diag_round25_afcm_mechanics.py). Uses only packages already installed in the project-local R
# (survival, splines, jsonlite); mgcv is not installed, so this is NOT a comparison with the original AFCM code.
# Usage: Rscript scripts/diag_round25_afcm_check.R <dir>
suppressPackageStartupMessages({library(survival); library(splines); library(jsonlite)})
d <- commandArgs(trailingOnly = TRUE)[1]
X <- as.matrix(read.csv(file.path(d, "design.csv"), header = FALSE))
yd <- as.matrix(read.csv(file.path(d, "y_d.csv"), header = FALSE))
beta <- scan(file.path(d, "beta.csv"), quiet = TRUE)
S <- as.matrix(read.csv(file.path(d, "penalty.csv"), header = FALSE))
inp <- fromJSON(file.path(d, "r_inputs.json"))
y <- yd[, 1]; ev <- yd[, 2]; E <- sum(ev)

# 1. Efron log partial likelihood at the Python coefficients: survival::coxph with no iteration
fit <- suppressWarnings(coxph(Surv(y, ev) ~ X, init = beta, ties = "efron", control = coxph.control(iter.max = 0)))
lp_r <- fit$loglik[1]

# 2. score vector at the Python coefficients, written out from the definition (Efron), and the penalised gradient
#    of the Python objective  -lp/E + beta' S beta
eta <- drop(X %*% beta); w <- exp(eta - max(eta))
U <- rep(0, ncol(X)); lp_own <- 0
for (u in unique(y[ev == 1])) {
  risk <- y >= u; dth <- (y == u) & (ev == 1); m <- sum(dth)
  s0 <- sum(w[risk]); s1 <- colSums(X[risk, , drop = FALSE] * w[risk])
  d0 <- sum(w[dth]);  d1 <- colSums(X[dth, , drop = FALSE] * w[dth])
  U <- U + colSums(X[dth, , drop = FALSE])
  lp_own <- lp_own + sum(eta[dth] - max(eta))
  for (l in 0:(m - 1)) {
    den <- s0 - l / m * d0
    U <- U - (s1 - l / m * d1) / den
    lp_own <- lp_own - log(den)
  }
}
grad <- -U / E + 2 * drop(S %*% beta)

# 3. cubic B-spline basis on the exported knots and points
kn <- scan(file.path(d, "basis_knots.csv"), quiet = TRUE)
xs <- scan(file.path(d, "basis_points.csv"), quiet = TRUE)
Bp <- as.matrix(read.csv(file.path(d, "basis_python.csv"), header = FALSE))
Br <- splineDesign(kn, xs, ord = 4)

write_json(list(status = "ok", lp_survival_coxph = lp_r, lp_own_efron = lp_own, lp_python = inp$lp_python,
                lp_abs_diff_coxph = abs(lp_r - inp$lp_python), lp_abs_diff_own = abs(lp_own - inp$lp_python),
                penalised_grad_max_abs = max(abs(grad)), penalised_grad_l2 = sqrt(sum(grad^2)),
                score_max_abs_unpenalised = max(abs(U / E)),
                basis_max_abs_diff = max(abs(Br - Bp)), basis_dim = dim(Br),
                versions = list(R = R.version.string, survival = as.character(packageVersion("survival")),
                                splines = as.character(packageVersion("splines")))),
           file.path(d, "r_result.json"), auto_unbox = TRUE, digits = NA)
