# D168 Stage A, step A1: the reference AFCM (Cui, Crainiceanu & Leroux 2021) fitted with mgcv, as specified by the
# author's public vignette (ecui1/AFCM, vignette_fcm.pdf, Part 1, non-periodic domain):
#   gam(y ~ z + ti(tmat, X, by = lmat, bs = c("cr","cr"), k = c(10,10), mc = c(FALSE,TRUE)),
#       weights = event, family = cox.ph())
# with one ti() term per functional channel, the quantile-transformed covariate (done by the caller from the TRAINING
# empirical CDF at each grid time, so that train and test use the same map), quadrature weights lmat on the grid, and
# method = "REML" stated explicitly (mgcv forces REML for the Cox family in any case).
# Usage: Rscript scripts/afcm_mgcv.R <workdir>
# Inputs in <workdir> (written by kanfae.models.afcm_mgcv): meta.json, tgrid.csv, lmat.csv, xq_tr_<d>.csv (n x G),
#   z_tr.csv, yd_tr.csv, optionally xq_te_<d>.csv and z_te.csv; xgrid.csv (points in [0,1] at which F_d is returned).
# Output: result.json (fit diagnostics, the surfaces F_d on tgrid x xgrid, linear predictors, z coefficients).
suppressPackageStartupMessages({library(mgcv); library(jsonlite)})
wd <- commandArgs(trailingOnly = TRUE)[1]
rd <- function(f) unname(as.matrix(read.csv(file.path(wd, f), header = FALSE)))
meta <- fromJSON(file.path(wd, "meta.json"))
D <- meta$D; kx <- meta$k_x; kt <- meta$k_t
tg <- drop(rd("tgrid.csv")); lw <- drop(rd("lmat.csv")); xgrid <- drop(rd("xgrid.csv"))
G <- length(tg)

frame <- function(part) {
  z <- rd(sprintf("z_%s.csv", part)); n <- nrow(z)
  df <- data.frame(row = seq_len(n))
  for (j in seq_len(ncol(z))) df[[paste0("z", j)]] <- z[, j]
  df$tmat <- I(matrix(tg, n, G, byrow = TRUE))
  df$lmat <- I(matrix(lw, n, G, byrow = TRUE))
  for (d in seq_len(D)) df[[paste0("X", d)]] <- I(rd(sprintf("xq_%s_%d.csv", part, d)))
  df
}
tr <- frame("tr")
yd <- rd("yd_tr.csv"); tr$y <- yd[, 1]; tr$event <- yd[, 2]
zn <- grep("^z[0-9]+$", names(tr), value = TRUE)
terms <- sprintf('ti(tmat, X%d, by = lmat, bs = c("cr", "cr"), k = c(%d, %d), mc = c(FALSE, TRUE))', seq_len(D), kt, kx)
fml <- as.formula(paste("y ~", paste(c(zn, terms), collapse = " + ")))

warn <- character(0)
t0 <- Sys.time()
fit <- tryCatch(withCallingHandlers(
  gam(fml, weights = event, data = tr, family = cox.ph(), method = "REML"),
  warning = function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") }),
  error = function(e) e)
secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
if (inherits(fit, "error")) {
  write_json(list(status = "failed", error = conditionMessage(fit), warnings = warn, seconds = secs),
             file.path(wd, "result.json"), auto_unbox = TRUE)
  quit(save = "no", status = 0)
}

# surfaces F_d(t, x) on tgrid x xgrid: one ti term evaluated at single points (lmat = 1, other terms irrelevant)
surf <- list()
for (d in seq_len(D)) {
  np <- G * length(xgrid)
  nd <- data.frame(row = seq_len(np))
  for (j in zn) nd[[j]] <- 0
  nd$tmat <- I(matrix(rep(tg, times = length(xgrid)), np, 1))
  nd$lmat <- I(matrix(1, np, 1))
  for (e in seq_len(D)) nd[[paste0("X", e)]] <- I(matrix(0.5, np, 1))
  nd[[paste0("X", d)]] <- I(matrix(rep(xgrid, each = G), np, 1))
  pt <- predict(fit, newdata = nd, type = "terms")
  col <- grep(sprintf("X%d\\)", d), colnames(pt))
  surf[[d]] <- matrix(pt[, col], G, length(xgrid))            # rows: t index, columns: x index
}
oi <- fit$outer.info
eta_te <- NULL
if (file.exists(file.path(wd, "z_te.csv"))) eta_te <- as.numeric(predict(fit, newdata = frame("te"), type = "link"))
write_json(list(status = "ok", seconds = secs, warnings = warn, method = fit$method,
                outer_conv = if (!is.null(oi$conv)) oi$conv else NA, outer_iter = if (!is.null(oi$iter)) oi$iter else NA,
                outer_grad_max_abs = if (!is.null(oi$grad)) max(abs(oi$grad)) else NA,
                sp = unname(fit$sp), edf_total = sum(fit$edf), reml = as.numeric(fit$gcv.ubre),
                n_coef = length(coef(fit)), gamma = unname(coef(fit)[zn]),
                surfaces = lapply(surf, function(m) unname(m)), xgrid = xgrid, tgrid = tg,
                eta_tr = as.numeric(predict(fit, newdata = frame("tr"), type = "link")), eta_te = eta_te,
                versions = list(R = R.version.string, mgcv = as.character(packageVersion("mgcv")))),
           file.path(wd, "result.json"), auto_unbox = TRUE, digits = NA)
