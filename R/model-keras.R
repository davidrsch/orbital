# orbital S3 method dispatch for all keras-based models.
# Implementation details are in model-keras-impl.R and model-keras-dag.R.

#' @rdname orbital
#' @method orbital keras.engine.sequential.Sequential
#' @section Keras backend:
#'   Three S3 methods are provided: one each for the legacy
#'   `keras.engine.sequential.Sequential`, the Keras 3
#'   `keras.src.models.sequential.Sequential`, and the functional-API
#'   `keras.src.models.functional.Functional` classes. All three dispatch to
#'   the same implementation and share the same set of supported layers; see
#'   `vignette("supported-models")` for the full dispatch table. Softmax layers
#'   must use `axis = -1`; `attention_mask` and LSTM/GRU masking are not
#'   supported and are rejected with an error.
#' @export
orbital.keras.engine.sequential.Sequential <- function(
  x,
  ...,
  mode = c("regression", "classification"),
  type = NULL,
  lvl = NULL,
  prefix = ".pred"
) {
  mode <- rlang::arg_match(mode)
  type <- default_type(type)
  extra <- list(...)
  orbital_keras_impl(
    x,
    mode = mode,
    type = type,
    lvl = lvl,
    prefix = prefix,
    feature_names = extra$feature_names
  )
}

#' @rdname orbital
#' @method orbital keras.src.models.sequential.Sequential
#' @export
orbital.keras.src.models.sequential.Sequential <- function(
  x,
  ...,
  mode = c("regression", "classification"),
  type = NULL,
  lvl = NULL,
  prefix = ".pred"
) {
  mode <- rlang::arg_match(mode)
  type <- default_type(type)
  extra <- list(...)
  orbital_keras_impl(
    x,
    mode = mode,
    type = type,
    lvl = lvl,
    prefix = prefix,
    feature_names = extra$feature_names
  )
}

#' @rdname orbital
#' @method orbital keras.src.models.functional.Functional
#' @export
orbital.keras.src.models.functional.Functional <- function(
  x,
  ...,
  mode = c("regression", "classification"),
  type = NULL,
  lvl = NULL,
  prefix = ".pred"
) {
  mode <- rlang::arg_match(mode)
  type <- default_type(type)
  extra <- list(...)
  orbital_keras_impl(
    x,
    mode = mode,
    type = type,
    lvl = lvl,
    prefix = prefix,
    feature_names = extra$feature_names
  )
}
