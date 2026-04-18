# Tree-based model implementations of estimate_orbital_size().
# Shared helper `estimate_tree_chars()` lives in estimate-size.R.

#' @rdname estimate_orbital_size
#' @export
estimate_orbital_size.xgb.Booster <- function(x, ...) {
  rlang::check_installed("xgboost")

  dump <- xgboost::xgb.dump(x, with_stats = FALSE)

  n_trees <- sum(startsWith(dump, "booster"))
  n_leaves <- sum(grepl("leaf=", dump, fixed = TRUE))
  n_internal <- length(dump) - n_trees - n_leaves

  # Sample internal lines to estimate average feature name length
  internal_idx <- which(grepl("<", dump, fixed = TRUE))
  if (length(internal_idx) > 50) {
    sample_idx <- internal_idx[seq(
      1,
      length(internal_idx),
      length.out = 50
    )]
  } else {
    sample_idx <- internal_idx
  }

  if (length(sample_idx) > 0) {
    features <- sub(".*\\[([^<]+)<.*", "\\1", dump[sample_idx])
    avg_feature_len <- mean(nchar(features))
  } else {
    avg_feature_len <- 5
  }

  estimate_tree_chars(n_trees, n_internal, avg_feature_len)
}

#' @rdname estimate_orbital_size
#' @export
estimate_orbital_size.lgb.Booster <- function(x, ...) {
  rlang::check_installed("lightgbm")

  model_json <- x$dump_model()
  model_info <- jsonlite::fromJSON(model_json)

  n_trees <- length(model_info$tree_info$num_leaves)
  # For a binary tree: n_internal = n_leaves - 1 per tree
  n_internal <- sum(model_info$tree_info$num_leaves - 1)

  feature_names <- model_info$feature_names
  if (length(feature_names) > 0) {
    avg_feature_len <- mean(nchar(feature_names))
  } else {
    avg_feature_len <- 5
  }

  estimate_tree_chars(n_trees, n_internal, avg_feature_len)
}

#' @rdname estimate_orbital_size
#' @export
estimate_orbital_size.ranger <- function(x, ...) {
  rlang::check_installed("ranger")

  n_trees <- x$num.trees

  # Count internal nodes: left child != 0 indicates internal node
  n_internal <- sum(vapply(
    x$forest$child.nodeIDs,
    function(tree) sum(tree[[1]] != 0),
    integer(1)
  ))

  feature_names <- x$forest$independent.variable.names
  if (length(feature_names) > 0) {
    avg_feature_len <- mean(nchar(feature_names))
  } else {
    avg_feature_len <- 5
  }

  estimate_tree_chars(n_trees, n_internal, avg_feature_len)
}

#' @rdname estimate_orbital_size
#' @export
estimate_orbital_size.randomForest <- function(x, ...) {
  rlang::check_installed("randomForest")

  n_trees <- x$ntree

  # Count internal nodes: leftDaughter != 0 indicates internal node
  n_internal <- sum(x$forest$leftDaughter != 0)

  # Get feature names from xlevels or fall back to generic names
  feature_names <- names(x$forest$xlevels)
  if (length(feature_names) > 0) {
    avg_feature_len <- mean(nchar(feature_names))
  } else {
    avg_feature_len <- 5
  }

  estimate_tree_chars(n_trees, n_internal, avg_feature_len)
}

#' @rdname estimate_orbital_size
#' @export
estimate_orbital_size.rpart <- function(x, ...) {
  n_internal <- sum(x$frame$var != "<leaf>")

  feature_names <- unique(x$frame$var[x$frame$var != "<leaf>"])
  if (length(feature_names) > 0) {
    avg_feature_len <- mean(nchar(feature_names))
  } else {
    avg_feature_len <- 5
  }

  estimate_tree_chars(1L, n_internal, avg_feature_len)
}

#' @rdname estimate_orbital_size
#' @export
estimate_orbital_size.constparty <- function(x, ...) {
  rlang::check_installed("partykit")

  n_total <- length(x)
  n_leaves <- length(partykit::nodeids(x, terminal = TRUE))
  n_internal <- n_total - n_leaves

  # Use all predictor variable names from the data
  feature_names <- names(x$data)
  # Remove response variable (first column is typically response)
  if (length(feature_names) > 1) {
    feature_names <- feature_names[-1]
  }
  if (length(feature_names) > 0) {
    avg_feature_len <- mean(nchar(feature_names))
  } else {
    avg_feature_len <- 5
  }

  estimate_tree_chars(1L, n_internal, avg_feature_len)
}

#' @rdname estimate_orbital_size
#' @export
estimate_orbital_size.catboost.Model <- function(x, ...) {
  rlang::check_installed("catboost")

  # Use tidypredict's parse_model which is fast (~4ms)
  pm <- tidypredict::parse_model(x)

  n_trees <- pm$general$niter
  # For symmetric/oblivious trees, all trees have the same number of leaves
  n_leaves_per_tree <- length(pm$trees[[1]])
  n_internal <- n_trees * (n_leaves_per_tree - 1)

  feature_names <- pm$general$feature_names
  if (length(feature_names) > 0) {
    avg_feature_len <- mean(nchar(feature_names))
  } else {
    avg_feature_len <- 5
  }

  estimate_tree_chars(n_trees, n_internal, avg_feature_len)
}
