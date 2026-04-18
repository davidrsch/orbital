# Shared helpers for Keras DAG layer handlers.
# Centralises tensor-weight extraction so every .k3_* handler validates
# weight arity with a consistent, user-readable error.

# Read a Keras layer's `get_weights()` list, verify it has at least
# `required` tensors, and return it. The `names` argument (optional) labels
# what each expected weight represents for the abort message.
#
# - `l`       : the live Keras layer object (must respond to $get_weights()).
# - `lname`   : layer name (for error context).
# - `required`: minimum number of weight tensors the handler needs.
# - `names`   : optional character vector of labels for the expected tensors
#              (e.g. c("kernel", "bias")); used only to decorate errors.
.k3_get_weights <- function(l, lname, required, names = NULL) {
  wts <- tryCatch(l$get_weights(), error = function(e) {
    cli::cli_abort(
      "Layer {.val {lname}}: failed to read weights ({conditionMessage(e)}).",
      call = NULL
    )
  })
  if (!is.list(wts)) {
    cli::cli_abort(
      "Layer {.val {lname}}: get_weights() did not return a list.",
      call = NULL
    )
  }
  n <- length(wts)
  if (n < required) {
    if (!is.null(names) && length(names) >= required) {
      expected <- paste(names[seq_len(required)], collapse = ", ")
      cli::cli_abort(
        c(
          "Layer {.val {lname}}: expected at least {required} weight tensor{?s} ({expected}), but got {n}.",
          i = "The layer may be unbuilt or have a non-standard weight layout."
        ),
        call = NULL
      )
    }
    cli::cli_abort(
      "Layer {.val {lname}}: expected at least {required} weight tensor{?s}, but got {n}.",
      call = NULL
    )
  }
  wts
}


# Read a Keras layer's `get_config()` and return the resulting list.
# On failure, emits `cli::cli_warn()` with the layer name so that fallbacks
# are visible to the caller rather than being silently substituted.
#
# - `l`       : the live Keras layer object (must respond to $get_config()).
# - `lname`   : layer name (for warning context).
# - `default` : value returned when get_config() errors. Defaults to list().
.k3_safe_get_config <- function(l, lname, default = list()) {
  tryCatch(
    l$get_config(),
    error = function(e) {
      cli::cli_warn(
        c(
          "Layer {.val {lname}}: get_config() failed ({conditionMessage(e)}).",
          i = "Falling back to defaults; inferred layer behaviour may differ from Keras."
        )
      )
      default
    }
  )
}


# Validate that a Keras Softmax / Activation(softmax) layer uses axis = -1.
# orbital's flat-column layout cannot express softmax over a non-feature
# axis, so any other axis value would silently produce wrong results.
#
# - `l`     : the live Keras layer object.
# - `lname` : layer name (for error context).
.check_softmax_axis <- function(l, lname) {
  cfg <- .k3_safe_get_config(l, lname)
  axis_raw <- cfg$axis
  if (is.null(axis_raw)) {
    return(invisible(NULL))
  }
  axis_vals <- tryCatch(
    as.integer(unlist(axis_raw)),
    error = function(e) NA_integer_
  )
  if (length(axis_vals) != 1L || is.na(axis_vals)) {
    cli::cli_abort(c(
      "Softmax layer {.val {lname}}: unsupported axis specification {.val {axis_raw}}.",
      i = "orbital only supports single-axis softmax on axis = -1 (the last/feature axis)."
    ))
  }
  if (!(axis_vals == -1L)) {
    cli::cli_abort(c(
      "Softmax layer {.val {lname}}: axis = {axis_vals} is not supported.",
      i = "orbital's flat-column layout only supports softmax over the last/feature axis (axis = -1).",
      i = "Rebuild the model with Softmax(axis = -1) or remove the explicit axis argument."
    ))
  }
  invisible(NULL)
}
