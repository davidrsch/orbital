# Handler functions for element-wise merge layers (Add, Multiply, Average,
# Maximum, Minimum, Subtract, Dot, Concatenate).
# Called by orbital_keras_dag_impl() in model-keras.R.

.k3_elementwise_merge <- function(
  lname,
  topo_map,
  expr_reg,
  state,
  layer_type,
  expr_fn
) {
  # Shared logic for element-wise merge layers (Add, Multiply, Average,
  # Maximum, Minimum). Each public handler is a thin wrapper that supplies
  # layer_type (for error messages) and expr_fn (expression builder).
  inbound <- topo_map[[lname]]
  if (is.null(inbound) || length(inbound) < 2L) {
    cli::cli_abort(c(
      "Keras {layer_type} layer {.val {lname}} must have at least 2 inbound inputs.",
      "i" = "got {length(inbound)}"
    ))
  }
  all_inbound_exprs <- lapply(inbound, function(nm) {
    get(nm, envir = expr_reg, inherits = FALSE)
  })
  widths <- lengths(all_inbound_exprs)
  if (length(unique(widths)) != 1L) {
    cli::cli_abort(c(
      "Keras {layer_type} layer {.val {lname}}: all inputs must have the same width.",
      "i" = "got: {paste(widths, collapse = ', ')}"
    ))
  }
  out_names <- paste0("orbital_", lname, "_h", seq_len(widths[1L]))
  out_exprs <- vapply(
    seq_len(widths[1L]),
    function(i) {
      terms <- vapply(
        all_inbound_exprs,
        function(e) backtick(e[i]),
        character(1L)
      )
      expr_fn(terms)
    },
    character(1L)
  )
  state$all_exprs[[lname]] <- stats::setNames(out_exprs, out_names)
  assign(lname, out_names, envir = expr_reg)
  invisible(NULL)
}

# Factory: produce a .k3_* handler for an element-wise merge layer
# whose output is a scalar combination of one term per input.
.k3_make_elementwise_merge_handler <- function(layer_type, expr_fn) {
  force(layer_type)
  force(expr_fn)
  function(
    l,
    lname,
    topo_map,
    expr_reg,
    state,
    weight_map,
    output_layer_names,
    last_dense
  ) {
    .k3_elementwise_merge(
      lname,
      topo_map,
      expr_reg,
      state,
      layer_type,
      expr_fn
    )
  }
}

# Table-driven declaration of element-wise merge handlers.
# Each row maps a Keras layer class to the combining expression.
.k3_elementwise_merge_specs <- list(
  Add = function(terms) paste0("(", paste(terms, collapse = " + "), ")"),
  Multiply = function(terms) paste0("(", paste(terms, collapse = " * "), ")"),
  Average = function(terms) {
    paste0(
      "((",
      paste(terms, collapse = " + "),
      ") / ",
      format_numeric(length(terms)),
      ")"
    )
  },
  Maximum = function(terms) {
    paste0("pmax(", paste(terms, collapse = ", "), ")")
  },
  Minimum = function(terms) {
    paste0("pmin(", paste(terms, collapse = ", "), ")")
  }
)

.k3_add <- .k3_make_elementwise_merge_handler(
  "Add",
  .k3_elementwise_merge_specs$Add
)
.k3_multiply <- .k3_make_elementwise_merge_handler(
  "Multiply",
  .k3_elementwise_merge_specs$Multiply
)
.k3_average <- .k3_make_elementwise_merge_handler(
  "Average",
  .k3_elementwise_merge_specs$Average
)
.k3_maximum <- .k3_make_elementwise_merge_handler(
  "Maximum",
  .k3_elementwise_merge_specs$Maximum
)
.k3_minimum <- .k3_make_elementwise_merge_handler(
  "Minimum",
  .k3_elementwise_merge_specs$Minimum
)


.k3_subtract <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Element-wise Subtract: first_input - second_input
  inbound <- topo_map[[lname]]
  if (is.null(inbound) || length(inbound) != 2L) {
    cli::cli_abort(c(
      "Keras Subtract layer {.val {lname}} requires exactly 2 inbound inputs.",
      "i" = "got {length(inbound)}"
    ))
  }
  exprs_a <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  exprs_b <- get(inbound[2L], envir = expr_reg, inherits = FALSE)
  if (length(exprs_a) != length(exprs_b)) {
    cli::cli_abort(c(
      "Keras Subtract layer {.val {lname}}: both inputs must have the same width.",
      "i" = "got: {length(exprs_a)} vs {length(exprs_b)}"
    ))
  }
  sub_names <- paste0("orbital_", lname, "_h", seq_len(length(exprs_a)))
  sub_exprs <- vapply(
    seq_along(exprs_a),
    function(i) paste0(backtick(exprs_a[i]), " - ", backtick(exprs_b[i])),
    character(1L)
  )
  state$all_exprs[[lname]] <- stats::setNames(sub_exprs, sub_names)
  assign(lname, sub_names, envir = expr_reg)
  invisible(NULL)
}


.k3_dot <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Dot product: sum of element-wise products of exactly 2 inputs (axes=-1)
  inbound <- topo_map[[lname]]
  if (is.null(inbound) || length(inbound) != 2L) {
    cli::cli_abort(
      "Keras Dot layer {.val {lname}} requires exactly 2 inbound inputs, got {length(inbound)}."
    )
  }
  exprs_a <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  exprs_b <- get(inbound[2L], envir = expr_reg, inherits = FALSE)
  if (length(exprs_a) != length(exprs_b)) {
    cli::cli_abort(
      "Keras Dot layer {.val {lname}}: both inputs must have the same width."
    )
  }
  cfg <- .k3_safe_get_config(l, lname)
  axes <- tryCatch(as.integer(cfg$axes), error = function(e) -1L)
  if (!all(axes %in% c(-1L, 1L))) {
    cli::cli_abort(
      "Keras Dot layer {.val {lname}}: only axes=-1 (feature-axis dot product) is supported."
    )
  }
  normalize <- isTRUE(cfg$normalize)
  dot_name <- paste0("orbital_", lname, "_dot")
  terms <- vapply(
    seq_along(exprs_a),
    function(i) {
      paste0("(", backtick(exprs_a[i]), " * ", backtick(exprs_b[i]), ")")
    },
    character(1L)
  )
  dot_expr <- paste0("(", paste(terms, collapse = " + "), ")")
  if (normalize) {
    # Cosine similarity: dot(a, b) / (||a||_2 * ||b||_2)
    sq_a <- vapply(
      exprs_a,
      function(e) paste0("(", backtick(e), ")^2"),
      character(1L)
    )
    sq_b <- vapply(
      exprs_b,
      function(e) paste0("(", backtick(e), ")^2"),
      character(1L)
    )
    norm_a <- paste0("sqrt(", paste(sq_a, collapse = " + "), ")")
    norm_b <- paste0("sqrt(", paste(sq_b, collapse = " + "), ")")
    denom_expr <- paste0("pmax(", norm_a, " * ", norm_b, ", 1e-12)")
    dot_expr <- paste0("(", dot_expr, " / ", denom_expr, ")")
  }
  state$all_exprs[[lname]] <- stats::setNames(dot_expr, dot_name)
  assign(lname, dot_name, envir = expr_reg)
  invisible(NULL)
}


.k3_concatenate <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Concatenate: merge expression-name vectors from all inbound branches
  inbound <- topo_map[[lname]]
  if (is.null(inbound) || length(inbound) < 2L) {
    cli::cli_abort(
      "Keras Concatenate layer {.val {lname}} must have at least 2 inbound inputs."
    )
  }
  combined_names <- unlist(lapply(inbound, function(nm) {
    get(nm, envir = expr_reg, inherits = FALSE)
  }))
  assign(lname, combined_names, envir = expr_reg)
  invisible(NULL)
}
