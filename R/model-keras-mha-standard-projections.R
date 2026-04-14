# Handler functions for MHA/GQA attention layers
# (GroupedQueryAttention, MultiHeadAttention).
# Called by orbital_keras_dag_impl() in model-keras.R.

# Internal: retrieve kernel and bias from an EinsumDense sub-layer of a Keras
# MHA/GQA layer.  Returns list(wq, wk, wv, wo) where each element is either
# list(kernel, bias) or NULL if the sub-layer cannot be accessed.
.mha_extract_sublayer_weights <- function(l, lname, use_bias) {
  .get_one <- function(attr_name) {
    sub <- tryCatch(
      reticulate::py_get_attr(l, attr_name),
      error = function(e) NULL
    )
    if (is.null(sub)) {
      return(NULL)
    }
    k <- tryCatch(
      as.array(reticulate::py_to_r(sub$kernel)),
      error = function(e) NULL
    )
    b <- if (use_bias) {
      tryCatch(
        as.array(reticulate::py_to_r(sub$bias)),
        error = function(e) NULL
      )
    } else {
      NULL
    }
    if (is.null(k)) {
      return(NULL)
    }
    list(kernel = k, bias = b)
  }
  list(
    wq = .get_one("_query_dense"),
    wk = .get_one("_key_dense"),
    wv = .get_one("_value_dense"),
    wo = .get_one("_output_dense")
  )
}

