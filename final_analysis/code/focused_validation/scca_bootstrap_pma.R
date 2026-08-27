#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  if (!requireNamespace("PMA", quietly = TRUE)) stop("PMA is not installed.")
})

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) return(default)
  args[[idx + 1]]
}

input_dir <- arg_value("--input-dir")
output_dir <- arg_value("--output-dir")
seed <- as.integer(arg_value("--seed", "1"))
components <- as.integer(strsplit(arg_value("--components", "1,2"), ",", fixed = TRUE)[[1]])
penaltyx_grid <- as.numeric(strsplit(arg_value("--penaltyx", "0.15,0.25,0.40"), ",", fixed = TRUE)[[1]])
penaltyz_grid <- as.numeric(strsplit(arg_value("--penaltyz", "0.15,0.25,0.40"), ",", fixed = TRUE)[[1]])
folds <- as.integer(arg_value("--folds", "5"))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

atomic_write_csv <- function(df, path) {
  tmp <- paste0(path, ".tmp")
  write.csv(df, tmp, row.names = FALSE)
  if (file.exists(path)) unlink(path)
  if (!file.rename(tmp, path)) stop(paste("Cannot write", path))
}

sd_eps <- 1e-12
safe_cor <- function(a, b) {
  if (length(a) < 3 || stats::sd(a) == 0 || stats::sd(b) == 0) return(NA_real_)
  as.numeric(stats::cor(a, b, use = "pairwise.complete.obs"))
}

make_folds <- function(n, k) sample(rep(seq_len(min(k, n)), length.out = n))
read_matrix <- function(path) data.matrix(read.csv(path, row.names = 1, check.names = FALSE))
nonzero_sd_columns <- function(m) {
  sds <- apply(m, 2, stats::sd)
  is.finite(sds) & sds > sd_eps
}
fit_pma <- function(x, z, k, px, pz) {
  PMA::CCA(x=x, z=z, typex="standard", typez="standard", K=k, penaltyx=px, penaltyz=pz, niter=100, trace=FALSE)
}
fit_pma_filtered <- function(x_train, z_train, k, px, pz, x_all_names, z_all_names) {
  keep_x <- nonzero_sd_columns(x_train)
  keep_z <- nonzero_sd_columns(z_train)
  if (sum(keep_x) < k || sum(keep_z) < k) stop("Too few non-zero-SD columns inside PMA fold-local filtering")
  model <- fit_pma(x_train[, keep_x, drop=FALSE], z_train[, keep_z, drop=FALSE], k, px, pz)
  u_small <- get_u(model)
  v_small <- get_v(model)
  u <- matrix(0, nrow=length(x_all_names), ncol=ncol(u_small), dimnames=list(x_all_names, NULL))
  v <- matrix(0, nrow=length(z_all_names), ncol=ncol(v_small), dimnames=list(z_all_names, NULL))
  u[which(keep_x), ] <- u_small
  v[which(keep_z), ] <- v_small
  list(model=model, u=u, v=v, keep_x=keep_x, keep_z=keep_z)
}
get_u <- function(model) {
  if (!is.null(model$u)) return(as.matrix(model$u))
  if (!is.null(model$ws)) return(as.matrix(model$ws))
  stop("No x loadings")
}
get_v <- function(model) {
  if (!is.null(model$v)) return(as.matrix(model$v))
  if (!is.null(model$ws.init.z)) return(as.matrix(model$ws.init.z))
  stop("No z loadings")
}
scores_for <- function(x, z, u, v) list(x=x %*% as.matrix(u), z=z %*% as.matrix(v))
loading_rows <- function(component, view, features, loadings) {
  data.frame(
    component=component,
    view=view,
    feature_id=features,
    loading=as.numeric(loadings),
    selected=abs(as.numeric(loadings)) > 1e-12,
    stringsAsFactors=FALSE,
    check.names=FALSE
  )
}

x <- read_matrix(file.path(input_dir, "microbiome.csv"))
z <- read_matrix(file.path(input_dir, "metabolome.csv"))
if (!identical(rownames(x), rownames(z))) stop("Sample order mismatch")
if (nrow(x) < 8) stop("Too few bootstrap rows for sCCA")
if (ncol(x) < 2 || ncol(z) < 2) stop("Too few features after bootstrap-local filtering")
if (any(!nonzero_sd_columns(x))) stop("Zero-SD microbiome columns reached PMA bootstrap input")
if (any(!nonzero_sd_columns(z))) stop("Zero-SD metabolome columns reached PMA bootstrap input")

fold_id <- make_folds(nrow(x), folds)
best <- NULL
best_score <- -Inf
best_sd <- Inf
best_selected <- Inf
tuning_rows <- list()

for (k in components) for (px in penaltyx_grid) for (pz in penaltyz_grid) {
  if (k > min(ncol(x), ncol(z), nrow(x) - 1)) next
  fs <- c()
  fsel <- c()
  for (fold in sort(unique(fold_id))) {
    train <- which(fold_id != fold)
    hold <- which(fold_id == fold)
    filtered <- fit_pma_filtered(x[train,,drop=FALSE], z[train,,drop=FALSE], k, px, pz, colnames(x), colnames(z))
    u <- filtered$u
    v <- filtered$v
    sc <- scores_for(x[hold,,drop=FALSE], z[hold,,drop=FALSE], u, v)
    cors <- vapply(seq_len(k), function(cmp) safe_cor(sc$x[,cmp], sc$z[,cmp]), numeric(1))
    score <- mean(abs(cors), na.rm=TRUE)
    selected <- sum(abs(u) > 0) + sum(abs(v) > 0)
    fs <- c(fs, score)
    fsel <- c(fsel, selected)
    tuning_rows[[length(tuning_rows)+1]] <- data.frame(
      n_components=k, penaltyx=px, penaltyz=pz, fold=fold,
      tuning_fold_abs_cross_view_correlation=score,
      selected_feature_count=selected,
      fold_local_microbiome_features_removed=sum(!filtered$keep_x),
      fold_local_metabolome_features_removed=sum(!filtered$keep_z),
      row_type="fold",
      selected_candidate=FALSE
    )
  }
  mean_score <- mean(fs, na.rm=TRUE)
  sd_score <- stats::sd(fs, na.rm=TRUE)
  if (!is.finite(sd_score)) sd_score <- 0
  mean_selected <- mean(fsel, na.rm=TRUE)
  if (is.finite(mean_score) &&
      (mean_score > best_score ||
       (mean_score == best_score && sd_score < best_sd) ||
       (mean_score == best_score && sd_score == best_sd && mean_selected < best_selected))) {
    best <- list(k=k, px=px, pz=pz)
    best_score <- mean_score
    best_sd <- sd_score
    best_selected <- mean_selected
  }
  tuning_rows[[length(tuning_rows)+1]] <- data.frame(
    n_components=k, penaltyx=px, penaltyz=pz, fold="mean",
    tuning_fold_abs_cross_view_correlation=mean_score,
    selected_feature_count=NA,
    fold_local_microbiome_features_removed=NA,
    fold_local_metabolome_features_removed=NA,
    row_type="candidate_mean",
    selected_candidate=FALSE
  )
}
if (is.null(best)) stop("No valid candidate")
for (j in seq_along(tuning_rows)) {
  if (tuning_rows[[j]]$n_components[1] == best$k &&
      tuning_rows[[j]]$penaltyx[1] == best$px &&
      tuning_rows[[j]]$penaltyz[1] == best$pz) {
    tuning_rows[[j]]$selected_candidate <- TRUE
  }
}

final_filtered <- fit_pma_filtered(x, z, best$k, best$px, best$pz, colnames(x), colnames(z))
final <- final_filtered$model
u <- final_filtered$u
v <- final_filtered$v
sc <- scores_for(x, z, u, v)

summary_rows <- list()
micro_rows <- list()
met_rows <- list()
for (cmp in seq_len(best$k)) {
  summary_rows[[length(summary_rows)+1]] <- data.frame(
    component=cmp,
    apparent_full_data_canonical_correlation=safe_cor(sc$x[,cmp], sc$z[,cmp]),
    n_samples=nrow(x),
    microbiome_features=ncol(x),
    metabolome_features=ncol(z),
    penaltyx=best$px,
    penaltyz=best$pz,
    tuning_score=best_score,
    tuning_score_sd=best_sd,
    selected_microbiome_feature_count=sum(abs(u[,cmp]) > 1e-12),
    selected_metabolome_feature_count=sum(abs(v[,cmp]) > 1e-12),
    phenotype_used_for_tuning=FALSE,
    phenotype_used_for_fit=FALSE,
    full_fit_microbiome_features_removed_for_zero_sd=sum(!final_filtered$keep_x),
    full_fit_metabolome_features_removed_for_zero_sd=sum(!final_filtered$keep_z),
    min_microbiome_sd_before_pma=min(apply(x, 2, stats::sd)),
    min_metabolome_sd_before_pma=min(apply(z, 2, stats::sd)),
    pma_internal_standardisation="typex=standard,typez=standard"
  )
  micro_rows[[length(micro_rows)+1]] <- loading_rows(cmp, "microbiome", colnames(x), u[,cmp])
  met_rows[[length(met_rows)+1]] <- loading_rows(cmp, "metabolome", colnames(z), v[,cmp])
}

atomic_write_csv(do.call(rbind, summary_rows), file.path(output_dir, "component_summary.csv"))
atomic_write_csv(do.call(rbind, micro_rows), file.path(output_dir, "microbiome_loadings.csv"))
atomic_write_csv(do.call(rbind, met_rows), file.path(output_dir, "metabolome_loadings.csv"))
atomic_write_csv(do.call(rbind, tuning_rows), file.path(output_dir, "tuning_summary.csv"))
