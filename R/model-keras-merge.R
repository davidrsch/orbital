# Handler functions for element-wise merge layers (Add, Multiply, Average,
# Maximum, Minimum, Subtract, Dot, Concatenate).
# Called by orbital_keras_dag_impl() in model-keras.R.


.k3_add <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # Element-wise Add: supports skip / residual connections (>=2 inputs)
  inbound <- topo_map[[lname]]
  if (is.null(inbound) || length(inbound) < 2L) {
    cli::cli_abort(
      "Keras Add layer {.val {lname}} must have at least 2 inbound inputs, got {length(inbound)}."
    )
  }
  all_inbound_exprs <- lapply(inbound, function(nm) {
    get(nm, envir = expr_reg, inherits = FALSE)
  })
  widths <- lengths(all_inbound_exprs)
  if (length(unique(widths)) != 1L) {
    cli::cli_abort(
      "Keras Add layer {.val {lname}}: all inputs must have the same width (got: {paste(widths, collapse = ', ')})."
    )
  }
  add_names <- paste0("orbital_", lname, "_h", seq_len(widths[1L]))
  add_exprs <- vapply(
    seq_len(widths[1L]),
    function(i) {
      terms <- vapply(
        all_inbound_exprs,
        function(e) backtick(e[i]),
        character(1L)
      )
      paste0("(", paste(terms, collapse = " + "), ")")
    },
    character(1L)
  )
  state$all_exprs[[lname]] <- stats::setNames(add_exprs, add_names)
  assign(lname, add_names, envir = expr_reg)
  invisible(NULL)
}


.k3_multiply <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # Element-wise Multiply: element-wise product of >=2 inputs
  inbound <- topo_map[[lname]]
  if (is.null(inbound) || length(inbound) < 2L) {
    cli::cli_abort(
      "Keras Multiply layer {.val {lname}} must have at least 2 inbound inputs, got {length(inbound)}."
    )
  }
  all_inbound_exprs <- lapply(inbound, function(nm) {
    get(nm, envir = expr_reg, inherits = FALSE)
  })
  widths <- lengths(all_inbound_exprs)
  if (length(unique(widths)) != 1L) {
    cli::cli_abort(
      "Keras Multiply layer {.val {lname}}: all inputs must have the same width (got: {paste(widths, collapse = ', ')})."
    )
  }
  mul_names <- paste0("orbital_", lname, "_h", seq_len(widths[1L]))
  mul_exprs <- vapply(
    seq_len(widths[1L]),
    function(i) {
      terms <- vapply(
        all_inbound_exprs,
        function(e) backtick(e[i]),
        character(1L)
      )
      paste0("(", paste(terms, collapse = " * "), ")")
    },
    character(1L)
  )
  state$all_exprs[[lname]] <- stats::setNames(mul_exprs, mul_names)
  assign(lname, mul_names, envir = expr_reg)
  invisible(NULL)
}


.k3_average <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # Element-wise Average: mean of >=2 inputs
  inbound <- topo_map[[lname]]
  if (is.null(inbound) || length(inbound) < 2L) {
    cli::cli_abort(
      "Keras Average layer {.val {lname}} must have at least 2 inbound inputs, got {length(inbound)}."
    )
  }
  all_inbound_exprs <- lapply(inbound, function(nm) {
    get(nm, envir = expr_reg, inherits = FALSE)
  })
  widths <- lengths(all_inbound_exprs)
  if (length(unique(widths)) != 1L) {
    cli::cli_abort(
      "Keras Average layer {.val {lname}}: all inputs must have the same width (got: {paste(widths, collapse = ', ')})."
    )
  }
  n_inputs <- length(inbound)
  avg_names <- paste0("orbital_", lname, "_h", seq_len(widths[1L]))
  avg_exprs <- vapply(
    seq_len(widths[1L]),
    function(i) {
      terms <- vapply(
        all_inbound_exprs,
        function(e) backtick(e[i]),
        character(1L)
      )
      paste0(
        "((",
        paste(terms, collapse = " + "),
        ") / ",
        format_numeric(n_inputs),
        ")"
      )
    },
    character(1L)
  )
  state$all_exprs[[lname]] <- stats::setNames(avg_exprs, avg_names)
  assign(lname, avg_names, envir = expr_reg)
  invisible(NULL)
}


.k3_maximum <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # Element-wise Maximum: per-element max over >=2 inputs
  inbound <- topo_map[[lname]]
  if (is.null(inbound) || length(inbound) < 2L) {
    cli::cli_abort(
      "Keras Maximum layer {.val {lname}} must have at least 2 inbound inputs, got {length(inbound)}."
    )
  }
  all_inbound_exprs <- lapply(inbound, function(nm) {
    get(nm, envir = expr_reg, inherits = FALSE)
  })
  widths <- lengths(all_inbound_exprs)
  if (length(unique(widths)) != 1L) {
    cli::cli_abort(
      "Keras Maximum layer {.val {lname}}: all inputs must have the same width (got: {paste(widths, collapse = ', ')})."
    )
  }
  max_names <- paste0("orbital_", lname, "_h", seq_len(widths[1L]))
  max_exprs <- vapply(
    seq_len(widths[1L]),
    function(i) {
      terms <- vapply(
        all_inbound_exprs,
        function(e) backtick(e[i]),
        character(1L)
      )
      paste0("pmax(", paste(terms, collapse = ", "), ")")
    },
    character(1L)
  )
  state$all_exprs[[lname]] <- stats::setNames(max_exprs, max_names)
  assign(lname, max_names, envir = expr_reg)
  invisible(NULL)
}


.k3_minimum <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # Element-wise Minimum: per-element min over >=2 inputs
  inbound <- topo_map[[lname]]
  if (is.null(inbound) || length(inbound) < 2L) {
    cli::cli_abort(
      "Keras Minimum layer {.val {lname}} must have at least 2 inbound inputs, got {length(inbound)}."
    )
  }
  all_inbound_exprs <- lapply(inbound, function(nm) {
    get(nm, envir = expr_reg, inherits = FALSE)
  })
  widths <- lengths(all_inbound_exprs)
  if (length(unique(widths)) != 1L) {
    cli::cli_abort(
      "Keras Minimum layer {.val {lname}}: all inputs must have the same width (got: {paste(widths, collapse = ', ')})."
    )
  }
  min_names <- paste0("orbital_", lname, "_h", seq_len(widths[1L]))
  min_exprs <- vapply(
    seq_len(widths[1L]),
    function(i) {
      terms <- vapply(
        all_inbound_exprs,
        function(e) backtick(e[i]),
        character(1L)
      )
      paste0("pmin(", paste(terms, collapse = ", "), ")")
    },
    character(1L)
  )
  state$all_exprs[[lname]] <- stats::setNames(min_exprs, min_names)
  assign(lname, min_names, envir = expr_reg)
  invisible(NULL)
}


.k3_subtract <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # Element-wise Subtract: first_input - second_input
  inbound <- topo_map[[lname]]
  if (is.null(inbound) || length(inbound) != 2L) {
    cli::cli_abort(
      "Keras Subtract layer {.val {lname}} requires exactly 2 inbound inputs, got {length(inbound)}."
    )
  }
  exprs_a <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  exprs_b <- get(inbound[2L], envir = expr_reg, inherits = FALSE)
  if (length(exprs_a) != length(exprs_b)) {
    cli::cli_abort(
      "Keras Subtract layer {.val {lname}}: both inputs must have the same width (got: {length(exprs_a)} vs {length(exprs_b)})."
    )
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


.k3_dot <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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
  dot_name <- paste0("orbital_", lname, "_dot")
  terms <- vapply(
    seq_along(exprs_a),
    function(i) {
      paste0("(", backtick(exprs_a[i]), " * ", backtick(exprs_b[i]), ")")
    },
    character(1L)
  )
  dot_expr <- paste0("(", paste(terms, collapse = " + "), ")")
  state$all_exprs[[lname]] <- stats::setNames(dot_expr, dot_name)
  assign(lname, dot_name, envir = expr_reg)
  invisible(NULL)
}


.k3_concatenate <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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

