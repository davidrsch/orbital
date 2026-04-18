# Linear-model implementations of estimate_orbital_size().
# Shared helper `estimate_linear_chars()` lives in estimate-size.R.

#' @rdname estimate_orbital_size
#' @export
estimate_orbital_size.glm <- function(x, ...) {
  coefs <- stats::coef(x)
  n_coefs <- length(coefs)

  # Exclude intercept from feature name calculation
  feature_names <- names(coefs)
  feature_names <- feature_names[feature_names != "(Intercept)"]
  if (length(feature_names) > 0) {
    avg_feature_len <- mean(nchar(feature_names))
  } else {
    avg_feature_len <- 5
  }

  estimate_linear_chars(n_coefs, avg_feature_len)
}

#' @rdname estimate_orbital_size
#' @export
estimate_orbital_size.lm <- estimate_orbital_size.glm

#' @rdname estimate_orbital_size
#' @export
estimate_orbital_size.glmnet <- function(x, ..., penalty = NULL) {
  rlang::check_installed("glmnet")

  if (is.null(penalty)) {
    if (length(x$lambda) != 1) {
      cli::cli_abort(
        c(
          "glmnet model has multiple penalty values.",
          "i" = "Specify a single {.arg penalty} value."
        )
      )
    }
    penalty <- x$lambda
  }

  coefs <- stats::coef(x, s = penalty)
  coef_values <- as.numeric(coefs)
  coef_names <- rownames(coefs)

  # Count non-zero coefficients
  non_zero_idx <- which(coef_values != 0)
  n_coefs <- length(non_zero_idx)

  # Only count features with non-zero coefficients
  feature_names <- coef_names[non_zero_idx]
  feature_names <- feature_names[feature_names != "(Intercept)"]

  if (length(feature_names) > 0) {
    avg_feature_len <- mean(nchar(feature_names))
  } else {
    avg_feature_len <- 5
  }

  estimate_linear_chars(n_coefs, avg_feature_len)
}

#' @rdname estimate_orbital_size
#' @export
estimate_orbital_size.earth <- function(x, ...) {
  rlang::check_installed("earth")

  coefs <- stats::coef(x)
  n_coefs <- length(coefs)

  # Earth coefficient names include hinge functions like "h(disp-145)"
  feature_names <- names(coefs)
  feature_names <- feature_names[feature_names != "(Intercept)"]
  if (length(feature_names) > 0) {
    avg_feature_len <- mean(nchar(feature_names))
  } else {
    avg_feature_len <- 5
  }

  estimate_linear_chars(n_coefs, avg_feature_len)
}
