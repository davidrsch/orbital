# Handler function for the CategoryEncoding preprocessing layer.
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

.k3_categoryencoding <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # CategoryEncoding: one-hot or multi-hot encoding of integer category indices.
  # For single input: output_k = as.integer(x == k) for k in 0..num_tokens-1.
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  cfg_l <- tryCatch(l$get_config(), error = function(e) list())
  num_tokens <- as.integer(cfg_l$num_tokens)
  output_mode <- tolower(as.character(cfg_l$output_mode))
  if (output_mode == "count") {
    cli::cli_abort(c(
      "CategoryEncoding layer {.val {lname}}: output_mode = \"count\" is not supported.",
      "i" = "Only \"one_hot\" and \"multi_hot\" are supported by orbital."
    ))
  }
  unit_names <- paste0(
    "orbital_catenc_",
    lname,
    "_h",
    seq_len(num_tokens)
  )
  enc_exprs <- vapply(
    seq_len(num_tokens) - 1L,
    function(k) {
      if (output_mode == "multi_hot" && length(in_exprs) > 1L) {
        parts <- vapply(
          in_exprs,
          function(col) paste0(backtick(col), " == ", k, "L"),
          character(1L)
        )
        paste0("as.integer(", paste(parts, collapse = " | "), ")")
      } else {
        xe <- backtick(in_exprs[[1L]])
        paste0("as.integer(", xe, " == ", k, "L)")
      }
    },
    character(1L)
  )
  state$all_exprs[[lname]] <- stats::setNames(enc_exprs, unit_names)
  assign(lname, unit_names, envir = expr_reg)
  invisible(NULL)
}
