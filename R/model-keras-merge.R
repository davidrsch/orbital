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


.k3_add <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Element-wise Add: supports skip / residual connections (>=2 inputs)
  .k3_elementwise_merge(
    lname,
    topo_map,
    expr_reg,
    state,
    "Add",
    function(terms) paste0("(", paste(terms, collapse = " + "), ")")
  )
}


.k3_multiply <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Element-wise Multiply: element-wise product of >=2 inputs
  .k3_elementwise_merge(
    lname,
    topo_map,
    expr_reg,
    state,
    "Multiply",
    function(terms) paste0("(", paste(terms, collapse = " * "), ")")
  )
}


.k3_average <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Element-wise Average: mean of >=2 inputs
  .k3_elementwise_merge(
    lname,
    topo_map,
    expr_reg,
    state,
    "Average",
    function(terms) {
      paste0(
        "((",
        paste(terms, collapse = " + "),
        ") / ",
        format_numeric(length(terms)),
        ")"
      )
    }
  )
}


.k3_maximum <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Element-wise Maximum: per-element max over >=2 inputs
  .k3_elementwise_merge(
    lname,
    topo_map,
    expr_reg,
    state,
    "Maximum",
    function(terms) paste0("pmax(", paste(terms, collapse = ", "), ")")
  )
}


.k3_minimum <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Element-wise Minimum: per-element min over >=2 inputs
  .k3_elementwise_merge(
    lname,
    topo_map,
    expr_reg,
    state,
    "Minimum",
    function(terms) paste0("pmin(", paste(terms, collapse = ", "), ")")
  )
}


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
  cfg <- tryCatch(l$get_config(), error = function(e) list())
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
