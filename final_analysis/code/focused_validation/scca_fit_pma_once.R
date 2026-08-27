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
k <- as.integer(arg_value("--components", "1"))
px <- as.numeric(arg_value("--penaltyx", "0.4"))
pz <- as.numeric(arg_value("--penaltyz", "0.4"))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

atomic_write_csv <- function(df, path) {
  tmp <- paste0(path, ".tmp")
  write.csv(df, tmp, row.names = FALSE)
  if (file.exists(path)) unlink(path)
  if (!file.rename(tmp, path)) stop(paste("Cannot write", path))
}
safe_cor <- function(a, b) {
  if (length(a) < 3 || stats::sd(a) == 0 || stats::sd(b) == 0) return(NA_real_)
  as.numeric(stats::cor(a, b, use = "pairwise.complete.obs"))
}
read_matrix <- function(path) data.matrix(read.csv(path, row.names = 1, check.names = FALSE))
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

x <- read_matrix(file.path(input_dir, "microbiome.csv"))
z <- read_matrix(file.path(input_dir, "metabolome.csv"))
if (!identical(rownames(x), rownames(z))) stop("Sample order mismatch")
if (k > min(nrow(x) - 1, ncol(x), ncol(z))) stop("Requested K exceeds matrix rank support")
if (any(apply(x, 2, stats::sd) <= 1e-12)) stop("Zero or near-zero-SD microbiome column reached PMA")
if (any(apply(z, 2, stats::sd) <= 1e-12)) stop("Zero or near-zero-SD metabolome column reached PMA")

model <- PMA::CCA(x=x, z=z, typex="standard", typez="standard", K=k, penaltyx=px, penaltyz=pz, niter=100, trace=FALSE)
u <- get_u(model)
v <- get_v(model)
x_scores <- x %*% u
z_scores <- z %*% v

summary_rows <- list()
micro_rows <- list()
met_rows <- list()
for (cmp in seq_len(k)) {
  summary_rows[[length(summary_rows)+1]] <- data.frame(
    component=cmp,
    apparent_canonical_correlation=safe_cor(x_scores[,cmp], z_scores[,cmp]),
    n_samples=nrow(x),
    microbiome_features=ncol(x),
    metabolome_features=ncol(z),
    penaltyx=px,
    penaltyz=pz,
    pma_internal_standardisation="typex=standard,typez=standard",
    stringsAsFactors=FALSE
  )
  micro_rows[[length(micro_rows)+1]] <- data.frame(component=cmp, view="microbiome", feature_id=colnames(x), loading=as.numeric(u[,cmp]), selected=abs(as.numeric(u[,cmp])) > 1e-12, stringsAsFactors=FALSE, check.names=FALSE)
  met_rows[[length(met_rows)+1]] <- data.frame(component=cmp, view="metabolome", feature_id=colnames(z), loading=as.numeric(v[,cmp]), selected=abs(as.numeric(v[,cmp])) > 1e-12, stringsAsFactors=FALSE, check.names=FALSE)
}
atomic_write_csv(do.call(rbind, summary_rows), file.path(output_dir, "summary.csv"))
atomic_write_csv(do.call(rbind, micro_rows), file.path(output_dir, "microbiome_loadings.csv"))
atomic_write_csv(do.call(rbind, met_rows), file.path(output_dir, "metabolome_loadings.csv"))
atomic_write_csv(data.frame(sample_key=rownames(x), x_scores, check.names=FALSE), file.path(output_dir, "x_scores.csv"))
atomic_write_csv(data.frame(sample_key=rownames(x), z_scores, check.names=FALSE), file.path(output_dir, "z_scores.csv"))
