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
penalties <- as.numeric(strsplit(arg_value("--penalties", "0.15,0.25,0.40"), ",", fixed = TRUE)[[1]])
folds <- as.integer(arg_value("--folds", "5"))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)
manifest <- read.csv(file.path(input_dir, "cohort_manifest.csv"), stringsAsFactors = FALSE)
atomic_write_csv <- function(df, path) { tmp <- paste0(path, ".tmp"); write.csv(df, tmp, row.names = FALSE); if (file.exists(path)) unlink(path); if (!file.rename(tmp, path)) stop(paste("Cannot write", path)) }
safe_cor <- function(a, b) { if (stats::sd(a) == 0 || stats::sd(b) == 0) return(NA_real_); as.numeric(stats::cor(a, b, use = "pairwise.complete.obs")) }
make_folds <- function(n, k) sample(rep(seq_len(min(k, n)), length.out = n))
read_matrix <- function(path) data.matrix(read.csv(path, row.names = 1, check.names = FALSE))
fit_pma <- function(x, z, k, px, pz) PMA::CCA(x=x, z=z, typex="standard", typez="standard", K=k, penaltyx=px, penaltyz=pz, niter=100, trace=FALSE)
get_u <- function(model) { if (!is.null(model$u)) return(as.matrix(model$u)); if (!is.null(model$ws)) return(as.matrix(model$ws)); stop("No x loadings") }
get_v <- function(model) { if (!is.null(model$v)) return(as.matrix(model$v)); if (!is.null(model$ws.init.z)) return(as.matrix(model$ws.init.z)); stop("No z loadings") }
scores_for <- function(x, z, u, v) list(x=x %*% as.matrix(u), z=z %*% as.matrix(v))
summary_rows <- list(); micro_rows <- list(); met_rows <- list(); tuning_rows <- list()
loading_rows <- function(disease, cohort, component, view, features, loadings) data.frame(disease=disease, cohort=cohort, component=component, view=view, feature_id=features, feature_name=sub("^(mic__|met__)", "", features), loading=as.numeric(loadings), selected=abs(as.numeric(loadings)) > 1e-12, stringsAsFactors=FALSE, check.names=FALSE)
for (i in seq_len(nrow(manifest))) {
  row <- manifest[i, ]; disease <- row$disease; cohort <- row$cohort
  x <- read_matrix(file.path(input_dir, row$microbiome_file)); z <- read_matrix(file.path(input_dir, row$metabolome_file))
  if (!identical(rownames(x), rownames(z))) stop(paste("Sample order mismatch", disease, cohort))
  fold_id <- make_folds(nrow(x), folds); best <- NULL; best_score <- -Inf; best_sd <- Inf; best_selected <- Inf
  for (k in components) for (px in penalties) for (pz in penalties) {
    if (k > min(ncol(x), ncol(z), nrow(x)-1)) next
    fs <- c(); fsel <- c()
    for (fold in sort(unique(fold_id))) {
      train <- which(fold_id != fold); hold <- which(fold_id == fold)
      model <- fit_pma(x[train,,drop=FALSE], z[train,,drop=FALSE], k, px, pz)
      u <- get_u(model); v <- get_v(model); sc <- scores_for(x[hold,,drop=FALSE], z[hold,,drop=FALSE], u, v)
      cors <- vapply(seq_len(k), function(cmp) safe_cor(sc$x[,cmp], sc$z[,cmp]), numeric(1))
      score <- mean(abs(cors), na.rm=TRUE); selected <- sum(abs(u)>0) + sum(abs(v)>0)
      fs <- c(fs, score); fsel <- c(fsel, selected)
      tuning_rows[[length(tuning_rows)+1]] <- data.frame(disease=disease, cohort=cohort, n_components=k, penaltyx=px, penaltyz=pz, fold=fold, tuning_fold_abs_cross_view_correlation=score, tuning_fold_abs_cross_view_correlation_sd=NA, selected_feature_count=selected, mean_selected_feature_count_across_training_folds=NA, tuning_criterion="unsupervised inner tuning-fold mean absolute canonical correlation", phenotype_used_for_tuning=FALSE, pma_function="PMA::CCA", row_type="fold", selected_candidate=FALSE)
    }
    mean_score <- mean(fs, na.rm=TRUE); sd_score <- stats::sd(fs, na.rm=TRUE); if (!is.finite(sd_score)) sd_score <- 0
    mean_selected <- mean(fsel, na.rm=TRUE)
    if (is.finite(mean_score) && (mean_score > best_score || (mean_score == best_score && sd_score < best_sd) || (mean_score == best_score && sd_score == best_sd && mean_selected < best_selected))) {
      best <- list(k=k, px=px, pz=pz); best_score <- mean_score; best_sd <- sd_score; best_selected <- mean_selected
    }
    tuning_rows[[length(tuning_rows)+1]] <- data.frame(disease=disease, cohort=cohort, n_components=k, penaltyx=px, penaltyz=pz, fold="mean", tuning_fold_abs_cross_view_correlation=mean_score, tuning_fold_abs_cross_view_correlation_sd=sd_score, selected_feature_count=NA, mean_selected_feature_count_across_training_folds=mean_selected, tuning_criterion="unsupervised inner tuning-fold mean absolute canonical correlation", phenotype_used_for_tuning=FALSE, pma_function="PMA::CCA", row_type="candidate_mean", selected_candidate=FALSE)
  }
  if (is.null(best)) stop(paste("No valid candidate", disease, cohort))
  for (j in seq_along(tuning_rows)) if (tuning_rows[[j]]$disease[1] == disease && tuning_rows[[j]]$cohort[1] == cohort && tuning_rows[[j]]$n_components[1] == best$k && tuning_rows[[j]]$penaltyx[1] == best$px && tuning_rows[[j]]$penaltyz[1] == best$pz) tuning_rows[[j]]$selected_candidate <- TRUE
  final <- fit_pma(x, z, best$k, best$px, best$pz); u <- get_u(final); v <- get_v(final); sc <- scores_for(x, z, u, v)
  colnames(sc$x) <- seq_len(ncol(sc$x)); colnames(sc$z) <- seq_len(ncol(sc$z))
  atomic_write_csv(data.frame(sample_id=rownames(x), sc$x, check.names=FALSE), file.path(output_dir, paste0(disease, "_", cohort, "_x_scores.csv")))
  atomic_write_csv(data.frame(sample_id=rownames(x), sc$z, check.names=FALSE), file.path(output_dir, paste0(disease, "_", cohort, "_z_scores.csv")))
  for (cmp in seq_len(best$k)) {
    summary_rows[[length(summary_rows)+1]] <- data.frame(disease=disease, cohort=cohort, component=cmp, apparent_full_data_canonical_correlation=safe_cor(sc$x[,cmp], sc$z[,cmp]), n_samples=nrow(x), microbiome_features=ncol(x), metabolome_features=ncol(z), penaltyx=best$px, penaltyz=best$pz, pma_function="PMA::CCA", phenotype_used_for_fit=FALSE)
    micro_rows[[length(micro_rows)+1]] <- loading_rows(disease, cohort, cmp, "microbiome", colnames(x), u[,cmp])
    met_rows[[length(met_rows)+1]] <- loading_rows(disease, cohort, cmp, "metabolome", colnames(z), v[,cmp])
  }
}
atomic_write_csv(do.call(rbind, summary_rows), file.path(output_dir, "stage4c_scca_component_summary.csv"))
atomic_write_csv(do.call(rbind, micro_rows), file.path(output_dir, "stage4c_scca_microbiome_loadings.csv"))
atomic_write_csv(do.call(rbind, met_rows), file.path(output_dir, "stage4c_scca_metabolome_loadings.csv"))
atomic_write_csv(do.call(rbind, tuning_rows), file.path(output_dir, "stage4c_scca_tuning_summary.csv"))
