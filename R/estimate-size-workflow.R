# Workflow / recipe / tailor implementations of estimate_orbital_size()
# and the internal step / adjustment character-count generics.

#' @rdname estimate_orbital_size
#' @export
estimate_orbital_size.recipe <- function(x, ...) {
  rlang::check_installed("recipes")

  if (!recipes::fully_trained(x)) {
    cli::cli_abort("recipe must be fully trained.")
  }

  total_chars <- 0L

  for (step in x$steps) {
    if (step$skip) {
      next
    }
    step_chars <- estimate_step_chars(step)
    total_chars <- total_chars + step_chars
  }

  total_chars
}

#' @rdname estimate_orbital_size
#' @export
estimate_orbital_size.workflow <- function(x, ...) {
  rlang::check_installed("workflows")

  if (!workflows::is_trained_workflow(x)) {
    cli::cli_abort("{.arg x} must be a fully trained {.cls workflow}.")
  }

  total_chars <- 0L

  # Estimate recipe contribution
  preprocessor <- workflows::extract_preprocessor(x)
  if (inherits(preprocessor, "recipe")) {
    recipe_fit <- workflows::extract_recipe(x)
    total_chars <- total_chars + estimate_orbital_size(recipe_fit, ...)
  }

  # Estimate model contribution
  model_fit <- workflows::extract_fit_parsnip(x)
  model_chars <- tryCatch(
    estimate_orbital_size(model_fit$fit, ...),
    error = function(e) {
      # Fall back to 0 if model type not supported
      0L
    }
  )
  total_chars <- total_chars + model_chars

  # Estimate tailor contribution
  if ("tailor" %in% names(x$post$actions)) {
    tailor_fit <- workflows::extract_tailor(x)
    total_chars <- total_chars + estimate_orbital_size(tailor_fit, ...)
  }

  total_chars
}

#' @rdname estimate_orbital_size
#' @export
estimate_orbital_size.tailor <- function(x, ...) {
  rlang::check_installed("tailor")

  if (is.null(x$columns)) {
    cli::cli_abort("{.arg x} must be a fitted {.cls tailor}.")
  }

  total_chars <- 0L

  for (adj in x$adjustments) {
    adj_chars <- estimate_adj_chars(adj)
    total_chars <- total_chars + adj_chars
  }

  total_chars
}

# Step estimation generic and methods ----------------------------------------

# Internal generic for estimating step character counts
estimate_step_chars <- function(x, ...) {
  UseMethod("estimate_step_chars")
}

# Default: estimate based on number of columns affected
# Most steps produce ~40 chars per column as a rough baseline
estimate_step_chars.default <- function(x, ...) {
  n_cols <- length(x$columns %||% 0L)
  as.integer(n_cols * 40)
}

# Adjustment estimation generic and methods -----------------------------------

# Internal generic for estimating adjustment character counts
estimate_adj_chars <- function(x, ...) {
  UseMethod("estimate_adj_chars")
}

# Default: most adjustments produce ~80 chars for a case_when expression
estimate_adj_chars.default <- function(x, ...) {
  80L
}
