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
