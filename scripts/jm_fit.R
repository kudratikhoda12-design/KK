# Round 24 (D163): Bayesian joint model comparator (JMbayes2) for one landmark partition.
# Usage: Rscript scripts/jm_fit.R <input.json> <output.json>
# Input (written by scripts/run_round24.py): training subjects' full longitudinal histories and survival from the
# landmark (delayed entry at t_L), validation and test subjects' histories up to t_L, the prediction times, the
# longitudinal forms to fit, the MCMC settings and the seed.
# Model (fixed in docs/round24_preregistration.md), fitted once per form:
#   longitudinal  for each marker, form "lin": lme(value ~ t, random = ~ t | id)            (intercept and slope)
#                                  form "ns2": lme(value ~ ns(t, 2), random = ~ t | id)  (natural cubic spline mean
#                                  trajectory, subject-specific intercept and slope; spline random effects did not
#                                  converge in lme on the development partitions)
#   survival      coxph(Surv(entry, time, event) ~ baseline covariates), entry = t_L (left truncation)
#   association   current value of every marker (JMbayes2 default)
# Output per form: conditional survival S(t_L + s | T > t_L, history up to t_L) for every validation and test subject
# on the prediction grid (or the recorded error if a fit failed).
suppressPackageStartupMessages({library(JMbayes2); library(survival); library(nlme); library(splines); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
inp <- fromJSON(args[1])
ch <- inp$channels                      # sanitised marker names m1..mD
zn <- inp$z_names                       # sanitised covariate names z1..zp
Ltr <- as.data.frame(inp$train_long)
Str <- as.data.frame(inp$train_surv)
fml_s <- if (length(zn)) reformulate(zn, response = "Surv(entry, time, event)") else Surv(entry, time, event) ~ 1
cox <- coxph(fml_s, data = Str, x = TRUE, model = TRUE)
FIX <- list(lin = "t", ns2 = "ns(t, 2)")
times <- inp$t_L + inp$pred_grid

predict_set <- function(fit, L, Sd) {
  # dynamic prediction: history up to t_L, event-free at t_L (JMbayes2 convention: survival time = t_L, event = 0).
  # Entry just before t_L: the model was fitted with delayed entry at t_L (random effects of the landmark cohort), so
  # the posterior uses the history and that random-effects distribution without a second survival-to-t_L factor.
  nd <- merge(as.data.frame(L), as.data.frame(Sd)[, c("id", zn), drop = FALSE], by = "id")
  nd$time <- inp$t_L; nd$event <- 0; nd$entry <- inp$t_L - 1e-6
  nd <- nd[order(nd$id, nd$t), ]
  p <- as.data.frame(predict(fit, newdata = nd, process = "event", times = times, return_newdata = TRUE,
                              cores = 1L, seed = inp$seed))   # one core: results independent of the machine
  ids <- as.data.frame(Sd)$id
  S <- matrix(NA_real_, length(ids), length(times))
  for (i in seq_along(ids)) {
    pi <- p[p$id == ids[i], ]
    for (j in seq_along(times)) {
      k <- which(abs(pi$t - times[j]) < 1e-9)
      if (length(k)) S[i, j] <- 1 - pi$pred_CIF[k[1]]
    }
  }
  list(ids = ids, S = S)
}

res <- list()
for (form in inp$forms) {
  t0 <- Sys.time()
  out <- tryCatch({
    lmes <- lapply(ch, function(m) {
      # the same data frame for every marker (JMbayes2 requirement); visits without this marker are excluded
      lme(as.formula(paste(m, "~", FIX[[form]])), random = ~ t | id, data = Ltr,
          na.action = na.exclude, control = lmeControl(opEstim = "optim", maxIter = 200, msMaxIter = 200))
    })
    fit <- jm(cox, lmes, time_var = "t", id_var = "id", n_chains = inp$n_chains, cores = 1L,
              n_iter = inp$n_iter, n_burnin = inp$n_burnin, seed = inp$seed)
    rh <- tryCatch(unname(fit$statistics$Rhat$alphas[, 1]), error = function(e) NULL)
    list(status = "ok", val = predict_set(fit, inp$val_long, inp$val_surv),
         test = predict_set(fit, inp$test_long, inp$test_surv), rhat_alphas = rh)
  }, error = function(e) list(status = "failed", error = conditionMessage(e)))
  out$seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  res[[form]] <- out
}
write_json(list(status = "ok", grid = inp$pred_grid, forms = res,
                versions = list(R = R.version.string, JMbayes2 = as.character(packageVersion("JMbayes2")),
                                nlme = as.character(packageVersion("nlme")), survival = as.character(packageVersion("survival")))),
           args[2], auto_unbox = TRUE, digits = NA, na = "null")
