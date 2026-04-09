# Handlers for layer types that are DAG pass-throughs or data-reshaping ops.
# (Dropout variants, Permute, RepeatVector, Embedding).
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

.k3_dropout_passthru <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Inference-transparent pass-through layers
  inbound <- topo_map[[lname]]
  if (
    !is.null(inbound) &&
      length(inbound) >= 1L &&
      exists(inbound[1L], envir = expr_reg, inherits = FALSE)
  ) {
    assign(
      lname,
      get(inbound[1L], envir = expr_reg, inherits = FALSE),
      envir = expr_reg
    )
  }
  invisible(NULL)
}


.k3_permute <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Permute: reorder the axes (dimensions) of the input tensor.
  # In the tabular/1D context the input is [T × C] (time-step major flat).
  # cfg$dims is 1-indexed (Keras convention) over the non-batch axes.
  # For a 2-D input the only supported permutations are (1,2) (no-op) and (2,1).
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  in_names <- in_exprs # column-name vector

  cfg <- tryCatch(l$get_config(), error = function(e) list())
  dims <- tryCatch(as.integer(unlist(cfg$dims)), error = function(e) {
    c(1L, 2L)
  })

  if (length(dims) == 2L && all(dims == c(1L, 2L))) {
    # No-op permutation (1,2): pass through unchanged.
    perm_names <- in_names
  } else if (length(dims) == 2L && all(dims == c(2L, 1L))) {
    # Transpose: swap T and C.
    in_shape <- tryCatch(
      as.integer(unlist(l$input_shape)),
      error = function(e) NULL
    )
    C_feat <- if (!is.null(in_shape) && length(in_shape) >= 1L) {
      tail(in_shape[!is.na(in_shape)], 1L)
    } else {
      1L
    }
    T_in <- length(in_names) / C_feat
    # Transposed: iterate over C first then T (column-major → row-major swap)
    perm_names <- character(length(in_names))
    for (c_i in seq_len(C_feat)) {
      for (t_i in seq_len(T_in)) {
        perm_names[[(c_i - 1L) * T_in + t_i]] <- in_names[[
          (t_i - 1L) * C_feat + c_i
        ]]
      }
    }
  } else {
    cli::cli_abort(c(
      "Keras Permute layer {.val {lname}}: unsupported dims {paste(dims, collapse=',')}.",
      "i" = "Only (1,2) and (2,1) are supported."
    ))
  }
  perm_out_nms <- paste0(
    "orbital_permute_",
    lname,
    "_h",
    seq_along(perm_names)
  )
  state$all_exprs[[lname]] <- stats::setNames(perm_names, perm_out_nms)
  assign(lname, perm_out_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_repeatvector <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # RepeatVector: replicate the (flat) input feature vector n times.
  # cfg$n = repetition count; output shape = [n × C_in].
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

  cfg <- tryCatch(l$get_config(), error = function(e) list())
  n_rep <- tryCatch(as.integer(cfg[["n"]]), error = function(e) 1L)
  if (is.na(n_rep) || n_rep < 1L) {
    cli::cli_abort(
      "Keras RepeatVector layer {.val {lname}}: n must be a positive integer, got {n_rep}."
    )
  }

  out_exprs <- rep(in_exprs, times = n_rep)
  out_nms <- .orb_col_nms("repvec", lname, "h", length(out_exprs))
  state$all_exprs[[lname]] <- stats::setNames(out_exprs, out_nms)
  assign(lname, out_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_embedding <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Embedding: integer index lookup into a dense weight matrix.
  # Keras weight layout:
  #   embeddings : (vocab_size, embed_dim)
  # Input:  T flat integer columns (one token index per timestep, 0-indexed).
  # Output: T x embed_dim flat columns (time-step major).
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

  wts <- l$get_weights()
  emb_mat <- wts[[1L]] # (vocab_size, embed_dim)
  vocab_size <- dim(emb_mat)[1L]
  embed_dim <- dim(emb_mat)[2L]

  if (vocab_size > 50000L) {
    cli::cli_abort(c(
      "Keras Embedding layer {.val {lname}}: vocabulary size {vocab_size} exceeds the",
      " maximum supported limit of 50,000.",
      "x" = "Expanding this layer would generate >50,000 CASE WHEN branches per output column,",
      "     which most SQL engines cannot compile or execute.",
      "i" = "Consider replacing the Embedding layer with a pre-computed lookup table and",
      "     joining the index column against it inside the database instead."
    ))
  }

  if (vocab_size > 10000L) {
    cli::cli_warn(c(
      "Keras Embedding layer {.val {lname}}: vocabulary size {vocab_size} is large.",
      "i" = paste(
        "Generated case_when expressions may be very long.",
        "Consider reducing vocabulary size."
      )
    ))
  }

  T_in <- length(in_exprs) # one column per token position
  emb_nms <- character(0L)
  emb_exprs <- character(0L)
  for (t in seq_len(T_in)) {
    in_col <- in_exprs[t]
    for (d in seq_len(embed_dim)) {
      cases <- vapply(
        seq_len(vocab_size),
        function(i) {
          paste0(
            backtick(in_col),
            " == ",
            i - 1L,
            "L ~ ",
            format_numeric(emb_mat[i, d])
          )
        },
        character(1L)
      )
      expr_str <- paste0(
        "dplyr::case_when(",
        paste(cases, collapse = ", "),
        ", TRUE ~ NA_real_)"
      )
      nm <- paste0("orbital_emb_", lname, "_t", t, "_d", d)
      emb_nms <- c(emb_nms, nm)
      emb_exprs <- c(emb_exprs, expr_str)
    }
  }
  state$all_exprs[[lname]] <- stats::setNames(emb_exprs, emb_nms)
  assign(lname, emb_nms, envir = expr_reg)
  invisible(NULL)
}
