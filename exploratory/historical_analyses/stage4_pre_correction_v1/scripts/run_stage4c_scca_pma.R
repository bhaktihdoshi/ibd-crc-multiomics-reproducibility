#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("PMA", quietly = TRUE)) {
    stop("PMA is not installed. Install with: R -e 'install.packages(\"PMA\", repos=\"https://cloud.r-project.org\")'")
  }
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
components <- as.integer(strsplit(arg_value("--components", "1"), ",", fixed = TRUE)[[1]])
penalties <- as.numeric(strsplit(arg_value("--penalties", "0.15,0.25,0.40"), ",", fixed = TRUE)[[1]])
folds <- as.integer(arg_value("--folds", "5"))

if (is.null(input_dir) || is.null(output_dir)) {
  stop("Required arguments: --input-dir and --output-dir")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

cohort_manifest <- read.csv(file.path(input_dir, "cohort_manifest.csv"), stringsAsFactors = FALSE)

component_schema <- c(
  "disease", "cohort", "component", "canonical_correlation", "n_samples",
  "microbiome_features", "metabolome_features", "penaltyx", "penaltyz",
  "pma_function", "phenotype_used_for_fit"
)
loading_schema <- c(
  "disease", "cohort", "component", "view", "feature_id", "feature_name", "loading", "selected"
)
tuning_schema <- c(
  "disease", "cohort", "n_components", "penaltyx", "penaltyz", "fold",
  "heldout_abs_cross_view_correlation",
  "heldout_abs_cross_view_correlation_sd",
  "selected_feature_count",
  "mean_selected_feature_count_across_training_folds",
  "tuning_criterion",
  "phenotype_used_for_tuning",
  "pma_function",
  "row_type"
)

schema_row <- function(schema, values) {
  out <- as.list(rep(NA, length(schema)))
  names(out) <- schema
  for (nm in names(values)) out[[nm]] <- values[[nm]]
  as.data.frame(out, stringsAsFactors = FALSE, check.names = FALSE)
}

schema_bind <- function(rows, schema) {
  if (length(rows) == 0) {
    return(as.data.frame(setNames(as.list(rep(NA, length(schema))), schema), stringsAsFactors = FALSE, check.names = FALSE)[0, ])
  }
  normalized <- lapply(rows, function(row) {
    missing <- setdiff(schema, names(row))
    for (nm in missing) row[[nm]] <- NA
    row <- row[, schema, drop = FALSE]
    row
  })
  do.call(rbind, normalized)
}

atomic_write_csv <- function(df, path) {
  tmp <- paste0(path, ".tmp")
  write.csv(df, tmp, row.names = FALSE)
  if (file.exists(path)) unlink(path)
  ok <- file.rename(tmp, path)
  if (!ok) stop(paste("Failed to atomically move", tmp, "to", path))
}

make_loading_rows <- function(disease, cohort, component, view, feature_ids, loadings, tolerance = 1e-12) {
  loading_vector <- as.numeric(loadings)
  feature_ids <- as.character(feature_ids)
  if (length(loading_vector) != length(feature_ids)) {
    stop(paste(
      "Loading length mismatch for", disease, cohort, view, "component", component,
      "features=", length(feature_ids), "loadings=", length(loading_vector)
    ))
  }
  if (any(!is.finite(loading_vector))) {
    stop(paste("Non-finite loading value for", disease, cohort, view, "component", component))
  }
  data.frame(
    disease = disease,
    cohort = cohort,
    component = component,
    view = view,
    feature_id = feature_ids,
    feature_name = sub("^(mic__|met__)", "", feature_ids),
    loading = loading_vector,
    selected = abs(loading_vector) > tolerance,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

read_matrix <- function(path) {
  mat <- read.csv(path, row.names = 1, check.names = FALSE)
  data.matrix(mat)
}

fit_pma_cca <- function(x, z, k, penaltyx, penaltyz) {
  PMA::CCA(
    x = x,
    z = z,
    typex = "standard",
    typez = "standard",
    K = k,
    penaltyx = penaltyx,
    penaltyz = penaltyz,
    niter = 100,
    trace = FALSE
  )
}

get_u <- function(model) {
  if (!is.null(model$u)) return(as.matrix(model$u))
  if (!is.null(model$ws)) return(as.matrix(model$ws))
  stop("PMA::CCA output did not contain x-side canonical vectors.")
}

get_v <- function(model) {
  if (!is.null(model$v)) return(as.matrix(model$v))
  if (!is.null(model$ws.init.z)) return(as.matrix(model$ws.init.z))
  stop("PMA::CCA output did not contain z-side canonical vectors.")
}

canonical_scores <- function(x, z, u, v) {
  xs <- x %*% as.matrix(u)
  zs <- z %*% as.matrix(v)
  if (is.null(dim(xs))) xs <- matrix(xs, ncol = 1)
  if (is.null(dim(zs))) zs <- matrix(zs, ncol = 1)
  colnames(xs) <- seq_len(ncol(xs))
  colnames(zs) <- seq_len(ncol(zs))
  list(x = xs, z = zs)
}

safe_cor <- function(a, b) {
  if (stats::sd(a) == 0 || stats::sd(b) == 0) return(NA_real_)
  as.numeric(stats::cor(a, b, use = "pairwise.complete.obs"))
}

make_folds <- function(n, k) {
  k <- min(k, n)
  sample(rep(seq_len(k), length.out = n))
}

component_rows <- list()
micro_loading_rows <- list()
met_loading_rows <- list()
tuning_rows <- list()

for (row_idx in seq_len(nrow(cohort_manifest))) {
  row <- cohort_manifest[row_idx, ]
  disease <- row$disease
  cohort <- row$cohort
  x <- read_matrix(file.path(input_dir, row$microbiome_file))
  z <- read_matrix(file.path(input_dir, row$metabolome_file))
  if (nrow(x) != nrow(z)) stop(paste("Sample mismatch for", disease, cohort))
  if (!identical(rownames(x), rownames(z))) stop(paste("Sample order mismatch for", disease, cohort))
  fold_id <- make_folds(nrow(x), folds)
  best_score <- -Inf
  best_sd <- Inf
  best_selected <- Inf
  best_params <- NULL

  for (k in components) {
    if (k > min(ncol(x), ncol(z), nrow(x) - 1)) next
    for (penaltyx in penalties) {
      for (penaltyz in penalties) {
        fold_scores <- c()
        fold_selected <- c()
        for (fold in sort(unique(fold_id))) {
          train_idx <- which(fold_id != fold)
          holdout_idx <- which(fold_id == fold)
          model <- fit_pma_cca(x[train_idx, , drop = FALSE], z[train_idx, , drop = FALSE], k, penaltyx, penaltyz)
          u <- get_u(model)
          v <- get_v(model)
          scores <- canonical_scores(x[holdout_idx, , drop = FALSE], z[holdout_idx, , drop = FALSE], u, v)
          cors <- vapply(seq_len(k), function(component) safe_cor(scores$x[, component, drop = TRUE], scores$z[, component, drop = TRUE]), numeric(1))
          score <- mean(abs(cors), na.rm = TRUE)
          if (!is.finite(score)) score <- NA_real_
          selected <- sum(abs(u) > 0) + sum(abs(v) > 0)
          fold_scores <- c(fold_scores, score)
          fold_selected <- c(fold_selected, selected)
          tuning_rows[[length(tuning_rows) + 1]] <- schema_row(tuning_schema, list(
            disease = disease,
            cohort = cohort,
            n_components = k,
            penaltyx = penaltyx,
            penaltyz = penaltyz,
            fold = fold,
            heldout_abs_cross_view_correlation = score,
            selected_feature_count = selected,
            tuning_criterion = "unsupervised held-out mean absolute canonical correlation",
            phenotype_used_for_tuning = FALSE,
            pma_function = "PMA::CCA",
            row_type = "fold"
          ))
        }
        mean_score <- mean(fold_scores, na.rm = TRUE)
        sd_score <- stats::sd(fold_scores, na.rm = TRUE)
        mean_selected <- mean(fold_selected, na.rm = TRUE)
        if (!is.finite(sd_score)) sd_score <- 0
        if (
          is.finite(mean_score) &&
          (mean_score > best_score ||
             (mean_score == best_score && sd_score < best_sd) ||
             (mean_score == best_score && sd_score == best_sd && mean_selected < best_selected))
        ) {
          best_score <- mean_score
          best_sd <- sd_score
          best_selected <- mean_selected
          best_params <- list(k = k, penaltyx = penaltyx, penaltyz = penaltyz)
        }
        tuning_rows[[length(tuning_rows) + 1]] <- schema_row(tuning_schema, list(
          disease = disease,
          cohort = cohort,
          n_components = k,
          penaltyx = penaltyx,
          penaltyz = penaltyz,
          fold = "mean",
          heldout_abs_cross_view_correlation = mean_score,
          heldout_abs_cross_view_correlation_sd = sd_score,
          mean_selected_feature_count_across_training_folds = mean_selected,
          tuning_criterion = "unsupervised held-out mean absolute canonical correlation",
          phenotype_used_for_tuning = FALSE,
          pma_function = "PMA::CCA",
          row_type = "candidate_mean"
        ))
      }
    }
  }
  if (is.null(best_params)) stop(paste("No valid PMA sCCA candidate for", disease, cohort))
  final <- fit_pma_cca(x, z, best_params$k, best_params$penaltyx, best_params$penaltyz)
  u <- get_u(final)
  v <- get_v(final)
  scores <- canonical_scores(x, z, u, v)
  atomic_write_csv(data.frame(sample_id = rownames(x), scores$x, check.names = FALSE), file.path(output_dir, paste0(disease, "_", cohort, "_x_scores.csv")))
  atomic_write_csv(data.frame(sample_id = rownames(x), scores$z, check.names = FALSE), file.path(output_dir, paste0(disease, "_", cohort, "_z_scores.csv")))
  for (component in seq_len(best_params$k)) {
    corr <- safe_cor(scores$x[, component, drop = TRUE], scores$z[, component, drop = TRUE])
    component_rows[[length(component_rows) + 1]] <- schema_row(component_schema, list(
      disease = disease,
      cohort = cohort,
      component = component,
      canonical_correlation = corr,
      n_samples = nrow(x),
      microbiome_features = ncol(x),
      metabolome_features = ncol(z),
      penaltyx = best_params$penaltyx,
      penaltyz = best_params$penaltyz,
      pma_function = "PMA::CCA",
      phenotype_used_for_fit = FALSE
    ))
    micro_loading_rows[[length(micro_loading_rows) + 1]] <- make_loading_rows(
      disease, cohort, component, "microbiome", colnames(x), u[, component, drop = TRUE]
    )
    met_loading_rows[[length(met_loading_rows) + 1]] <- make_loading_rows(
      disease, cohort, component, "metabolome", colnames(z), v[, component, drop = TRUE]
    )
  }
}

atomic_write_csv(schema_bind(component_rows, component_schema), file.path(output_dir, "stage4c_scca_component_summary.csv"))
atomic_write_csv(schema_bind(micro_loading_rows, loading_schema), file.path(output_dir, "stage4c_scca_microbiome_loadings.csv"))
atomic_write_csv(schema_bind(met_loading_rows, loading_schema), file.path(output_dir, "stage4c_scca_metabolome_loadings.csv"))
atomic_write_csv(schema_bind(tuning_rows, tuning_schema), file.path(output_dir, "stage4c_scca_tuning_summary.csv"))
