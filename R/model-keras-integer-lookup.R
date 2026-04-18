# Handler function for the IntegerLookup preprocessing layer.
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

.k3_integerlookup <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # IntegerLookup: maps each integer input to its 0-based index in the vocabulary.
  # Vocabulary sourced from config (static) or first weight tensor (adapt()).
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  cfg_l <- .k3_safe_get_config(l, lname)
  vocab_raw <- cfg_l$vocabulary
  if (!is.null(vocab_raw) && length(vocab_raw) > 0L) {
    vocab <- as.integer(unlist(vocab_raw))
  } else {
    wts <- l$get_weights()
    vocab <- as.integer(wts[[1L]])
  }
  unit_names <- paste0(
    "orbital_intlook_",
    lname,
    "_h",
    seq_along(in_exprs)
  )
  lookup_exprs <- vapply(
    seq_along(in_exprs),
    function(j) {
      cases <- vapply(
        seq_along(vocab),
        function(k) {
          paste0(
            backtick(in_exprs[[j]]),
            " == ",
            vocab[k],
            "L ~ ",
            k - 1L,
            "L"
          )
        },
        character(1L)
      )
      paste0(
        "dplyr::case_when(",
        paste(cases, collapse = ", "),
        ", TRUE ~ NA_integer_)"
      )
    },
    character(1L)
  )
  state$all_exprs[[lname]] <- stats::setNames(lookup_exprs, unit_names)
  assign(lname, unit_names, envir = expr_reg)
  invisible(NULL)
}
