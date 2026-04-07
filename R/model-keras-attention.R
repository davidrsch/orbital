# Handler functions for Attention and AdditiveAttention layers.
# Called by orbital_keras_dag_impl() in model-keras.R.

.k3_attention <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Attention (Luong / dot-product): single-head dot-product soft-attention.
  # Inputs (inbound): [query, value] or [query, value, key].
  #   query shape: (T_q, D)  — flat layout T_q * D columns
  #   value shape: (T_v, D_v) — flat layout T_v * D_v columns
  #   key   shape: (T_k, D)  — defaults to value if not provided
  # use_scale = FALSE (default): no learned scale.
  # Algorithm:
  #   score[q, k] = sum_d query[q,d] * key[k,d]
  #   attn[q, k]  = softmax over k (max-stabilised)
  #   out[q, d_v] = sum_k attn[q,k] * value[k, d_v]
  inbound <- topo_map[[lname]]
  if (is.null(inbound) || length(inbound) < 2L) {
    cli::cli_abort(
      "Keras Attention layer {.val {lname}} requires at least 2 inbound inputs (query, value)."
    )
  }
  q_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  v_exprs <- get(inbound[2L], envir = expr_reg, inherits = FALSE)
  k_exprs <- if (length(inbound) >= 3L) {
    get(inbound[3L], envir = expr_reg, inherits = FALSE)
  } else {
    v_exprs # key defaults to value
  }

  cfg_att <- tryCatch(l$get_config(), error = function(e) list())
  use_scale_att <- tryCatch(
    as.logical(cfg_att$use_scale),
    error = function(e) FALSE
  )
  if (isTRUE(use_scale_att) && length(l$get_weights()) > 0L) {
    cli::cli_abort(
      "Keras Attention layer {.val {lname}}: use_scale=TRUE is not yet supported."
    )
  }

  # Infer shapes from input sizes.  Both query and key must have same depth D.
  # For self-attention query==key, D_q must equal D_k.
  # We infer T_q, D from q_exprs length, T_v and D_v from v_exprs,
  # T_k and D_k from k_exprs.  We need D_q == D_k for dot-product.
  # The simplest inference path: assume square sequence & same depth.
  in_shape_q <- tryCatch(
    as.integer(unlist(l$input_spec[[1L]]$shape)),
    error = function(e) NULL
  )
  in_shape_v <- tryCatch(
    as.integer(unlist(l$input_spec[[2L]]$shape)),
    error = function(e) NULL
  )

  D_q <- if (!is.null(in_shape_q) && length(in_shape_q) >= 1L) {
    tail(in_shape_q[!is.na(in_shape_q)], 1L)
  } else {
    # Fallback: assume square — length = T^2 or T * D; try sqrt
    as.integer(sqrt(length(q_exprs)))
  }
  T_q <- as.integer(length(q_exprs) / D_q)

  D_v <- if (!is.null(in_shape_v) && length(in_shape_v) >= 1L) {
    tail(in_shape_v[!is.na(in_shape_v)], 1L)
  } else {
    D_q
  }
  T_v <- as.integer(length(v_exprs) / D_v)
  T_k <- as.integer(length(k_exprs) / D_q)

  # Compute scores: score[q_i, k_j] = sum_d q_exprs[(q_i-1)*D_q + d] * k_exprs[(k_j-1)*D_q + d]
  # Then max-stabilised softmax over k for each q_i.
  att_out_exprs <- character(0L)
  att_out_nms <- character(0L)

  score_nms <- matrix(
    paste0(
      "orbital_att_",
      lname,
      "_sc_q",
      rep(seq_len(T_q), each = T_k),
      "_k",
      rep(seq_len(T_k), T_q)
    ),
    nrow = T_q,
    ncol = T_k
  )
  score_exprs <- matrix("", nrow = T_q, ncol = T_k)
  for (q_i in seq_len(T_q)) {
    for (k_j in seq_len(T_k)) {
      terms <- vapply(
        seq_len(D_q),
        function(d) {
          paste0(
            "(",
            backtick(q_exprs[(q_i - 1L) * D_q + d]),
            " * ",
            backtick(k_exprs[(k_j - 1L) * D_q + d]),
            ")"
          )
        },
        character(1L)
      )
      score_exprs[q_i, k_j] <- paste0(
        "(",
        paste(terms, collapse = " + "),
        ")"
      )
    }
  }

  # For each q_i, compute softmax over k dimension (max-stabilised).
  for (q_i in seq_len(T_q)) {
    sc_names <- score_nms[q_i, ]
    sc_expr_q <- score_exprs[q_i, ]

    # Register score intermediates
    for (k_j in seq_len(T_k)) {
      state$all_exprs[[lname]] <- c(
        state$all_exprs[[lname]],
        stats::setNames(sc_expr_q[k_j], sc_names[k_j])
      )
    }

    max_nm <- paste0("orbital_att_", lname, "_max_q", q_i)
    max_expr <- paste0(
      "pmax(",
      paste(backtick(sc_names), collapse = ", "),
      ")"
    )
    state$all_exprs[[lname]] <- c(
      state$all_exprs[[lname]],
      stats::setNames(max_expr, max_nm)
    )

    exp_nms <- paste0(
      "orbital_att_",
      lname,
      "_exp_q",
      q_i,
      "_k",
      seq_len(T_k)
    )
    exp_exprs <- vapply(
      sc_names,
      function(s) paste0("exp(", backtick(s), " - ", backtick(max_nm), ")"),
      character(1L)
    )
    for (k_j in seq_len(T_k)) {
      state$all_exprs[[lname]] <- c(
        state$all_exprs[[lname]],
        stats::setNames(exp_exprs[k_j], exp_nms[k_j])
      )
    }

    sum_nm <- paste0("orbital_att_", lname, "_sum_q", q_i)
    sum_expr <- paste0("(", paste(backtick(exp_nms), collapse = " + "), ")")
    state$all_exprs[[lname]] <- c(
      state$all_exprs[[lname]],
      stats::setNames(sum_expr, sum_nm)
    )

    attn_nms <- paste0(
      "orbital_att_",
      lname,
      "_attn_q",
      q_i,
      "_k",
      seq_len(T_k)
    )
    attn_exprs <- vapply(
      exp_nms,
      function(e) paste0("(", backtick(e), " / ", backtick(sum_nm), ")"),
      character(1L)
    )
    for (k_j in seq_len(T_k)) {
      state$all_exprs[[lname]] <- c(
        state$all_exprs[[lname]],
        stats::setNames(attn_exprs[k_j], attn_nms[k_j])
      )
    }

    # Output: weighted sum over value
    for (d_v in seq_len(D_v)) {
      terms <- vapply(
        seq_len(T_v),
        function(k_j) {
          paste0(
            "(",
            backtick(attn_nms[k_j]),
            " * ",
            backtick(v_exprs[(k_j - 1L) * D_v + d_v]),
            ")"
          )
        },
        character(1L)
      )
      out_expr <- paste0("(", paste(terms, collapse = " + "), ")")
      out_nm <- paste0("orbital_att_", lname, "_out_q", q_i, "_d", d_v)
      att_out_exprs <- c(att_out_exprs, out_expr)
      att_out_nms <- c(att_out_nms, out_nm)
      state$all_exprs[[lname]] <- c(
        state$all_exprs[[lname]],
        stats::setNames(out_expr, out_nm)
      )
    }
  }
  assign(lname, att_out_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_additiveattention <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # AdditiveAttention (Bahdanau attention):
  # score[q,k] = scale * sum_d tanh(query[q,d] + key[k,d])
  # where scale is 1.0 if use_scale=FALSE (the Keras default).
  # Softmax and output weighted-sum are identical to Attention.
  inbound <- topo_map[[lname]]
  if (is.null(inbound) || length(inbound) < 2L) {
    cli::cli_abort(
      "Keras AdditiveAttention layer {.val {lname}} requires at least 2 inbound inputs (query, value)."
    )
  }
  q_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  v_exprs <- get(inbound[2L], envir = expr_reg, inherits = FALSE)
  k_exprs <- if (length(inbound) >= 3L) {
    get(inbound[3L], envir = expr_reg, inherits = FALSE)
  } else {
    v_exprs
  }

  cfg_aa <- tryCatch(l$get_config(), error = function(e) list())
  use_scale_aa <- tryCatch(
    as.logical(cfg_aa$use_scale),
    error = function(e) FALSE
  )
  aa_scale <- if (isTRUE(use_scale_aa)) {
    wts_aa <- tryCatch(l$get_weights(), error = function(e) list())
    if (length(wts_aa) >= 1L) as.numeric(wts_aa[[1L]])[1L] else 1.0
  } else {
    1.0
  }

  in_shape_q_aa <- tryCatch(
    as.integer(unlist(l$input_spec[[1L]]$shape)),
    error = function(e) NULL
  )
  in_shape_v_aa <- tryCatch(
    as.integer(unlist(l$input_spec[[2L]]$shape)),
    error = function(e) NULL
  )
  D_q_aa <- if (!is.null(in_shape_q_aa) && length(in_shape_q_aa) >= 1L) {
    tail(in_shape_q_aa[!is.na(in_shape_q_aa)], 1L)
  } else {
    as.integer(sqrt(length(q_exprs)))
  }
  T_q_aa <- as.integer(length(q_exprs) / D_q_aa)
  D_v_aa <- if (!is.null(in_shape_v_aa) && length(in_shape_v_aa) >= 1L) {
    tail(in_shape_v_aa[!is.na(in_shape_v_aa)], 1L)
  } else {
    D_q_aa
  }
  T_v_aa <- as.integer(length(v_exprs) / D_v_aa)
  T_k_aa <- as.integer(length(k_exprs) / D_q_aa)

  att_out_exprs_aa <- character(0L)
  att_out_nms_aa <- character(0L)

  # Scores: sum_d tanh(q[d] + k[d]) * scale
  score_nms_aa <- matrix(
    paste0(
      "orbital_addatt_",
      lname,
      "_sc_q",
      rep(seq_len(T_q_aa), each = T_k_aa),
      "_k",
      rep(seq_len(T_k_aa), T_q_aa)
    ),
    nrow = T_q_aa,
    ncol = T_k_aa
  )
  score_exprs_aa <- matrix("", nrow = T_q_aa, ncol = T_k_aa)
  for (q_i in seq_len(T_q_aa)) {
    for (k_j in seq_len(T_k_aa)) {
      terms <- vapply(
        seq_len(D_q_aa),
        function(d) {
          paste0(
            "tanh(",
            backtick(q_exprs[(q_i - 1L) * D_q_aa + d]),
            " + ",
            backtick(k_exprs[(k_j - 1L) * D_q_aa + d]),
            ")"
          )
        },
        character(1L)
      )
      score_exprs_aa[q_i, k_j] <- paste0(
        "(",
        aa_scale,
        " * (",
        paste(terms, collapse = " + "),
        "))"
      )
    }
  }

  # Softmax + weighted output (identical structure to Attention)
  for (q_i in seq_len(T_q_aa)) {
    sc_names_aa <- score_nms_aa[q_i, ]
    sc_expr_q_aa <- score_exprs_aa[q_i, ]
    for (k_j in seq_len(T_k_aa)) {
      state$all_exprs[[lname]] <- c(
        state$all_exprs[[lname]],
        stats::setNames(sc_expr_q_aa[k_j], sc_names_aa[k_j])
      )
    }
    max_nm_aa <- paste0("orbital_addatt_", lname, "_max_q", q_i)
    max_expr_aa <- paste0(
      "pmax(",
      paste(backtick(sc_names_aa), collapse = ", "),
      ")"
    )
    state$all_exprs[[lname]] <- c(
      state$all_exprs[[lname]],
      stats::setNames(max_expr_aa, max_nm_aa)
    )
    exp_nms_aa <- paste0(
      "orbital_addatt_",
      lname,
      "_exp_q",
      q_i,
      "_k",
      seq_len(T_k_aa)
    )
    exp_exprs_aa <- vapply(
      sc_names_aa,
      function(s) {
        paste0("exp(", backtick(s), " - ", backtick(max_nm_aa), ")")
      },
      character(1L)
    )
    for (k_j in seq_len(T_k_aa)) {
      state$all_exprs[[lname]] <- c(
        state$all_exprs[[lname]],
        stats::setNames(exp_exprs_aa[k_j], exp_nms_aa[k_j])
      )
    }
    sum_nm_aa <- paste0("orbital_addatt_", lname, "_sum_q", q_i)
    sum_expr_aa <- paste0(
      "(",
      paste(backtick(exp_nms_aa), collapse = " + "),
      ")"
    )
    state$all_exprs[[lname]] <- c(
      state$all_exprs[[lname]],
      stats::setNames(sum_expr_aa, sum_nm_aa)
    )
    attn_nms_aa <- paste0(
      "orbital_addatt_",
      lname,
      "_attn_q",
      q_i,
      "_k",
      seq_len(T_k_aa)
    )
    attn_exprs_aa <- vapply(
      exp_nms_aa,
      function(e) paste0("(", backtick(e), " / ", backtick(sum_nm_aa), ")"),
      character(1L)
    )
    for (k_j in seq_len(T_k_aa)) {
      state$all_exprs[[lname]] <- c(
        state$all_exprs[[lname]],
        stats::setNames(attn_exprs_aa[k_j], attn_nms_aa[k_j])
      )
    }
    for (d_v in seq_len(D_v_aa)) {
      terms <- vapply(
        seq_len(T_v_aa),
        function(k_j) {
          paste0(
            "(",
            backtick(attn_nms_aa[k_j]),
            " * ",
            backtick(v_exprs[(k_j - 1L) * D_v_aa + d_v]),
            ")"
          )
        },
        character(1L)
      )
      out_expr_aa <- paste0("(", paste(terms, collapse = " + "), ")")
      out_nm_aa <- paste0(
        "orbital_addatt_",
        lname,
        "_out_q",
        q_i,
        "_d",
        d_v
      )
      att_out_exprs_aa <- c(att_out_exprs_aa, out_expr_aa)
      att_out_nms_aa <- c(att_out_nms_aa, out_nm_aa)
      state$all_exprs[[lname]] <- c(
        state$all_exprs[[lname]],
        stats::setNames(out_expr_aa, out_nm_aa)
      )
    }
  }
  assign(lname, att_out_nms_aa, envir = expr_reg)
  invisible(NULL)
}


