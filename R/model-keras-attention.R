# Handler functions for attention layers
# (Attention, AdditiveAttention, GroupedQueryAttention, MultiHeadAttention).
# Called by orbital_keras_dag_impl() in model-keras.R.


.k3_attention <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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


.k3_additiveattention <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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


.k3_groupedqueryattention <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # GroupedQueryAttention: GQA where K/V heads are shared across groups of Q heads.
  # num_query_groups (num_kv_heads) divides num_heads.
  # For query head h (1-indexed), KV group = ((h-1) %/% heads_per_group) + 1.
  # Weight axes accessed via EinsumDense sub-layers (same names as MHA).
  inbound <- topo_map[[lname]]
  q_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  kv_exprs <- get(
    inbound[min(2L, length(inbound))],
    envir = expr_reg,
    inherits = FALSE
  )

  cfg <- tryCatch(l$get_config(), error = function(e) list())
  num_heads <- as.integer(cfg$num_heads %||% 1L)
  head_dim <- as.integer(cfg$head_dim %||% 1L)
  num_kv <- as.integer(cfg$num_query_groups %||% num_heads)
  if (num_heads %% num_kv != 0L) {
    cli::cli_abort(c(
      "GroupedQueryAttention {.val {lname}}: num_heads ({num_heads}) must be divisible by num_query_groups ({num_kv})."
    ))
  }
  heads_per_group <- as.integer(num_heads / num_kv)
  use_bias <- isTRUE(as.logical(cfg$use_bias %||% TRUE))

  use_gate <- isTRUE(as.logical(cfg$use_gate %||% FALSE))
  if (use_gate) {
    cli::cli_abort(c(
      "GroupedQueryAttention {.val {lname}}: use_gate = TRUE is not supported by orbital.",
      "i" = "Only use_gate = FALSE is supported."
    ))
  }

  in_shape <- tryCatch(
    as.integer(unlist(l$input_shape[[1L]])),
    error = function(e) NULL
  )
  C_in <- if (!is.null(in_shape)) {
    tail(in_shape[!is.na(in_shape)], 1L)
  } else {
    1L
  }
  T_q <- as.integer(length(q_exprs) / C_in)
  T_kv <- as.integer(length(kv_exprs) / C_in)

  out_shape <- tryCatch(
    as.integer(unlist(l$output_shape)),
    error = function(e) NULL
  )
  C_out <- if (!is.null(out_shape)) {
    tail(out_shape[!is.na(out_shape)], 1L)
  } else {
    C_in
  }

  .gqa_wts <- function(attr_name) {
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

  wq <- .gqa_wts("_query_dense")
  wk <- .gqa_wts("_key_dense")
  wv <- .gqa_wts("_value_dense")
  wo <- .gqa_wts("_output_dense")

  if (is.null(wq) || is.null(wk) || is.null(wv) || is.null(wo)) {
    cli::cli_abort(c(
      "GroupedQueryAttention {.val {lname}}: could not extract sub-layer weights.",
      "i" = "Ensure keras3 >= 3.0 with reticulate access to _query_dense etc."
    ))
  }

  # Build accessor for a 3-D kernel (C_in, d2, d3) with possible dim swap.
  .gqa_proj_at <- function(k, d2, d3, name) {
    d <- dim(k)
    if (length(d) != 3L) {
      cli::cli_abort(
        "GQA {.val {lname}}: {name} kernel must be 3-D, got {length(d)}D."
      )
    }
    if (d[1L] != C_in) {
      cli::cli_abort(
        "GQA {.val {lname}}: {name} kernel dim1={d[1L]} != C_in={C_in}."
      )
    }
    if (d[2L] == d2 && d[3L] == d3) {
      function(c_i, i2, i3) k[c_i, i2, i3]
    } else if (d[2L] == d3 && d[3L] == d2) {
      function(c_i, i2, i3) k[c_i, i3, i2]
    } else {
      cli::cli_abort(
        "GQA {.val {lname}}: {name} kernel dims ({paste(d,collapse='x')}) unrecognised."
      )
    }
  }

  .gqa_bias2 <- function(b, d2, d3) {
    if (is.null(b)) {
      return(function(i2, i3) "0")
    }
    di <- dim(b)
    if (!is.null(di) && length(di) == 2L && di[1L] == d2 && di[2L] == d3) {
      function(i2, i3) format_numeric(b[i2, i3])
    } else if (
      !is.null(di) && length(di) == 2L && di[1L] == d3 && di[2L] == d2
    ) {
      function(i2, i3) format_numeric(b[i3, i2])
    } else {
      function(i2, i3) "0"
    }
  }

  wq_at <- .gqa_proj_at(wq$kernel, head_dim, num_heads, "Q")
  wk_at <- .gqa_proj_at(wk$kernel, head_dim, num_kv, "K")
  wv_at <- .gqa_proj_at(wv$kernel, head_dim, num_kv, "V")
  bq_at <- .gqa_bias2(wq$bias, head_dim, num_heads)
  bk_at <- .gqa_bias2(wk$bias, head_dim, num_kv)
  bv_at <- .gqa_bias2(wv$bias, head_dim, num_kv)

  wo_d <- dim(wo$kernel)
  if (length(wo_d) == 3L && wo_d[1L] == num_heads && wo_d[2L] == head_dim) {
    wo_at <- function(h, dv, do_) wo$kernel[h, dv, do_]
    C_out_w <- wo_d[3L]
  } else if (
    length(wo_d) == 3L && wo_d[1L] == C_out && wo_d[2L] == num_heads
  ) {
    wo_at <- function(h, dv, do_) wo$kernel[do_, h, dv]
    C_out_w <- wo_d[1L]
  } else {
    cli::cli_abort(
      "GQA {.val {lname}}: output kernel dims ({paste(wo_d, collapse='x')}) unrecognised."
    )
  }
  bo_at <- function(do_) {
    if (!is.null(wo$bias) && length(wo$bias) >= do_) {
      format_numeric(wo$bias[do_])
    } else {
      "0"
    }
  }

  q_at <- function(t, c) backtick(q_exprs[(t - 1L) * C_in + c])
  kv_at <- function(t, c) backtick(kv_exprs[(t - 1L) * C_in + c])

  gqa_nms <- character(0L)
  gqa_exprs <- character(0L)

  # Q projections: Q[q, h, d] = sum_c q_in[q,c] * W_q(c, d, h) + b_q(d, h)
  q_nm <- array(NA_character_, dim = c(T_q, num_heads, head_dim))
  for (q in seq_len(T_q)) {
    for (h in seq_len(num_heads)) {
      for (d in seq_len(head_dim)) {
        terms <- vapply(
          seq_len(C_in),
          function(c) {
            paste0(q_at(q, c), " * ", format_numeric(wq_at(c, d, h)))
          },
          character(1L)
        )
        nm <- paste0(
          "orbital_gqa_",
          lname,
          "_Q_q",
          q,
          "_h",
          h,
          "_d",
          d
        )
        gqa_nms <- c(gqa_nms, nm)
        gqa_exprs <- c(
          gqa_exprs,
          paste0(
            "(",
            paste(terms, collapse = " + "),
            " + ",
            bq_at(d, h),
            ")"
          )
        )
        q_nm[q, h, d] <- nm
      }
    }
  }

  # K projections: K[k, g, d] = sum_c kv_in[k,c] * W_k(c, d, g) + b_k(d, g)
  k_nm <- array(NA_character_, dim = c(T_kv, num_kv, head_dim))
  for (k in seq_len(T_kv)) {
    for (g in seq_len(num_kv)) {
      for (d in seq_len(head_dim)) {
        terms <- vapply(
          seq_len(C_in),
          function(c) {
            paste0(kv_at(k, c), " * ", format_numeric(wk_at(c, d, g)))
          },
          character(1L)
        )
        nm <- paste0(
          "orbital_gqa_",
          lname,
          "_K_k",
          k,
          "_g",
          g,
          "_d",
          d
        )
        gqa_nms <- c(gqa_nms, nm)
        gqa_exprs <- c(
          gqa_exprs,
          paste0(
            "(",
            paste(terms, collapse = " + "),
            " + ",
            bk_at(d, g),
            ")"
          )
        )
        k_nm[k, g, d] <- nm
      }
    }
  }

  # V projections: V[k, g, d_v] = sum_c kv_in[k,c] * W_v(c, d_v, g) + b_v(d_v, g)
  v_nm <- array(NA_character_, dim = c(T_kv, num_kv, head_dim))
  for (k in seq_len(T_kv)) {
    for (g in seq_len(num_kv)) {
      for (d_v in seq_len(head_dim)) {
        terms <- vapply(
          seq_len(C_in),
          function(c) {
            paste0(kv_at(k, c), " * ", format_numeric(wv_at(c, d_v, g)))
          },
          character(1L)
        )
        nm <- paste0(
          "orbital_gqa_",
          lname,
          "_V_k",
          k,
          "_g",
          g,
          "_dv",
          d_v
        )
        gqa_nms <- c(gqa_nms, nm)
        gqa_exprs <- c(
          gqa_exprs,
          paste0(
            "(",
            paste(terms, collapse = " + "),
            " + ",
            bv_at(d_v, g),
            ")"
          )
        )
        v_nm[k, g, d_v] <- nm
      }
    }
  }

  # Scaled dot-product attention with max-stabilised softmax.
  # Query head h uses KV group g = ((h-1) %/% heads_per_group) + 1.
  scale <- 1.0 / sqrt(as.numeric(head_dim))
  attn_nm <- array(NA_character_, dim = c(T_q, T_kv, num_heads))
  for (q in seq_len(T_q)) {
    for (h in seq_len(num_heads)) {
      g_h <- (h - 1L) %/% heads_per_group + 1L # KV group for this Q head

      score_exprs <- vapply(
        seq_len(T_kv),
        function(k) {
          dot_terms <- vapply(
            seq_len(head_dim),
            function(d) {
              paste0(
                backtick(q_nm[q, h, d]),
                " * ",
                backtick(k_nm[k, g_h, d])
              )
            },
            character(1L)
          )
          paste0(
            "((",
            paste(dot_terms, collapse = " + "),
            ") * ",
            format_numeric(scale),
            ")"
          )
        },
        character(1L)
      )

      max_nm <- paste0("orbital_gqa_", lname, "_smax_q", q, "_h", h)
      gqa_nms <- c(gqa_nms, max_nm)
      gqa_exprs <- c(
        gqa_exprs,
        paste0(
          "do.call(pmax, list(",
          paste(score_exprs, collapse = ", "),
          "))"
        )
      )

      exp_nms <- character(T_kv)
      for (k in seq_len(T_kv)) {
        exp_nm <- paste0(
          "orbital_gqa_",
          lname,
          "_exp_q",
          q,
          "_k",
          k,
          "_h",
          h
        )
        gqa_nms <- c(gqa_nms, exp_nm)
        gqa_exprs <- c(
          gqa_exprs,
          paste0("exp(", score_exprs[k], " - `", max_nm, "`)")
        )
        exp_nms[k] <- exp_nm
      }

      sumexp_nm <- paste0("orbital_gqa_", lname, "_sumexp_q", q, "_h", h)
      gqa_nms <- c(gqa_nms, sumexp_nm)
      gqa_exprs <- c(
        gqa_exprs,
        paste0("(", paste(backtick(exp_nms), collapse = " + "), ")")
      )

      for (k in seq_len(T_kv)) {
        a_nm <- paste0(
          "orbital_gqa_",
          lname,
          "_attn_q",
          q,
          "_k",
          k,
          "_h",
          h
        )
        gqa_nms <- c(gqa_nms, a_nm)
        gqa_exprs <- c(
          gqa_exprs,
          paste0(
            "(",
            backtick(exp_nms[k]),
            " / ",
            backtick(sumexp_nm),
            ")"
          )
        )
        attn_nm[q, k, h] <- a_nm
      }
    }
  }

  # Head outputs: hout[q, h, d_v] = sum_k attn[q,k,h] * V[k, g_h, d_v]
  hout_nm <- array(NA_character_, dim = c(T_q, num_heads, head_dim))
  for (q in seq_len(T_q)) {
    for (h in seq_len(num_heads)) {
      g_h <- (h - 1L) %/% heads_per_group + 1L
      for (d_v in seq_len(head_dim)) {
        terms <- vapply(
          seq_len(T_kv),
          function(k) {
            paste0(
              backtick(attn_nm[q, k, h]),
              " * ",
              backtick(v_nm[k, g_h, d_v])
            )
          },
          character(1L)
        )
        nm <- paste0(
          "orbital_gqa_",
          lname,
          "_hout_q",
          q,
          "_h",
          h,
          "_dv",
          d_v
        )
        gqa_nms <- c(gqa_nms, nm)
        gqa_exprs <- c(
          gqa_exprs,
          paste0("(", paste(terms, collapse = " + "), ")")
        )
        hout_nm[q, h, d_v] <- nm
      }
    }
  }

  # Output projection: out[q, d_out] = sum_h sum_dv hout[q,h,dv]*W_o(h,dv,d_out)+b_o
  out_nms <- character(0L)
  for (q in seq_len(T_q)) {
    for (d_out in seq_len(C_out_w)) {
      terms <- character(0L)
      for (h in seq_len(num_heads)) {
        for (d_v in seq_len(head_dim)) {
          terms <- c(
            terms,
            paste0(
              backtick(hout_nm[q, h, d_v]),
              " * ",
              format_numeric(wo_at(h, d_v, d_out))
            )
          )
        }
      }
      nm <- paste0("orbital_gqa_", lname, "_out_q", q, "_d", d_out)
      gqa_nms <- c(gqa_nms, nm)
      gqa_exprs <- c(
        gqa_exprs,
        paste0(
          "(",
          paste(terms, collapse = " + "),
          " + ",
          bo_at(d_out),
          ")"
        )
      )
      out_nms <- c(out_nms, nm)
    }
  }

  state$all_exprs[[lname]] <- stats::setNames(gqa_exprs, gqa_nms)
  assign(lname, out_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_multiheadattention <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # MultiHeadAttention: scaled dot-product self-attention or cross-attention.
  # Supports: fixed-length sequences; causal_mask = FALSE; use_bias = TRUE/FALSE.
  # Input layout: time-step major (T × C_in) flat vector.
  # Weight extraction via reticulate::py_get_attr on internal EinsumDense sub-layers.
  inbound <- topo_map[[lname]]
  q_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  kv_exprs <- get(
    inbound[min(2L, length(inbound))],
    envir = expr_reg,
    inherits = FALSE
  )

  cfg <- tryCatch(l$get_config(), error = function(e) list())
  num_heads <- as.integer(cfg$num_heads %||% 1L)
  key_dim <- as.integer(cfg$key_dim %||% 1L)
  value_dim <- as.integer(cfg$value_dim %||% key_dim)
  use_bias <- isTRUE(as.logical(cfg$use_bias %||% TRUE))

  in_shape <- tryCatch(
    as.integer(unlist(l$input_shape[[1L]])),
    error = function(e) NULL
  )
  C_in <- if (!is.null(in_shape)) {
    tail(in_shape[!is.na(in_shape)], 1L)
  } else {
    1L
  }
  T_q <- as.integer(length(q_exprs) / C_in)
  T_kv <- as.integer(length(kv_exprs) / C_in)

  out_shape <- tryCatch(
    as.integer(unlist(l$output_shape)),
    error = function(e) NULL
  )
  C_out <- if (!is.null(out_shape)) {
    tail(out_shape[!is.na(out_shape)], 1L)
  } else {
    C_in
  }

  # Retrieve kernel and bias from an EinsumDense sub-layer (handles _attr naming).
  .mha_wts <- function(attr_name) {
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

  wq <- .mha_wts("_query_dense")
  wk <- .mha_wts("_key_dense")
  wv <- .mha_wts("_value_dense")
  wo <- .mha_wts("_output_dense")

  if (is.null(wq) || is.null(wk) || is.null(wv) || is.null(wo)) {
    cli::cli_abort(
      "MultiHeadAttention {.val {lname}}: could not extract sub-layer projection weights.",
      "i" = "Ensure keras3 >= 3.0 with reticulate access to _query_dense, _key_dense, etc."
    )
  }

  # Determine weight accessor functions given that Keras3 EinsumDense stores
  # Q/K kernels as 3-D arrays.  Two known layouts:
  #   (a) (C_in, key_dim, num_heads)  — einsum "abc,ced->abde" with d=num_heads,e=key_dim
  #   (b) (C_in, num_heads, key_dim)  — alternative layout
  .make_proj_at <- function(k, dim2_size, dim3_size, name) {
    d <- dim(k)
    if (length(d) != 3L) {
      cli::cli_abort(
        "MHA {.val {lname}}: {name} kernel must be 3-D, got {length(d)}D."
      )
    }
    if (d[1L] != C_in) {
      cli::cli_abort(
        "MHA {.val {lname}}: {name} kernel dim1={d[1L]} does not match C_in={C_in}."
      )
    }
    if (d[2L] == dim2_size && d[3L] == dim3_size) {
      function(c_idx, idx2, idx3) k[c_idx, idx2, idx3]
    } else if (d[2L] == dim3_size && d[3L] == dim2_size) {
      function(c_idx, idx2, idx3) k[c_idx, idx3, idx2]
    } else {
      cli::cli_abort(
        "MHA {.val {lname}}: {name} kernel dims ({d}) cannot be reconciled with ",
        "expected ({C_in}, {dim2_size}, {dim3_size}) or transposed."
      )
    }
  }

  # Q/K: (C_in, ?, ?) with dims key_dim and num_heads in some order.
  # After .make_proj_at, call as wq_at(c, key_dim_idx, head_idx) → scalar.
  wq_at <- .make_proj_at(wq$kernel, key_dim, num_heads, "Q")
  wk_at <- .make_proj_at(wk$kernel, key_dim, num_heads, "K")
  # V: (C_in, ?, ?) with dims value_dim and num_heads.
  wv_at <- .make_proj_at(wv$kernel, value_dim, num_heads, "V")

  # Q bias accessor: shape (key_dim, num_heads) or (num_heads, key_dim).
  .make_bias_at2 <- function(b, d2, d3, name) {
    if (is.null(b)) {
      return(function(i2, i3) "0")
    }
    di <- dim(b)
    if (length(di) == 2L && di[1L] == d2 && di[2L] == d3) {
      function(i2, i3) format_numeric(b[i2, i3])
    } else if (length(di) == 2L && di[1L] == d3 && di[2L] == d2) {
      function(i2, i3) format_numeric(b[i3, i2])
    } else {
      function(i2, i3) "0"
    }
  }
  bq_at <- .make_bias_at2(wq$bias, key_dim, num_heads, "Q_bias")
  bk_at <- .make_bias_at2(wk$bias, key_dim, num_heads, "K_bias")
  bv_at <- .make_bias_at2(wv$bias, value_dim, num_heads, "V_bias")

  # Output projection: (num_heads, value_dim, C_out) or transposed.
  wo_d <- dim(wo$kernel)
  if (
    length(wo_d) == 3L && wo_d[1L] == num_heads && wo_d[2L] == value_dim
  ) {
    wo_at <- function(h, dv, do_) wo$kernel[h, dv, do_]
    C_out_w <- wo_d[3L]
  } else if (
    length(wo_d) == 3L && wo_d[1L] == C_out && wo_d[2L] == num_heads
  ) {
    # (C_out, num_heads, value_dim) layout
    wo_at <- function(h, dv, do_) wo$kernel[do_, h, dv]
    C_out_w <- wo_d[1L]
  } else {
    cli::cli_abort(
      "MHA {.val {lname}}: output kernel dims {wo_d} unrecognised."
    )
  }
  bo_at <- function(do_) {
    if (!is.null(wo$bias) && length(wo$bias) >= do_) {
      format_numeric(wo$bias[do_])
    } else {
      "0"
    }
  }

  # Helper: column name for an input at (timestep t, channel c)
  q_at <- function(t, c) backtick(q_exprs[(t - 1L) * C_in + c])
  kv_at <- function(t, c) backtick(kv_exprs[(t - 1L) * C_in + c])

  mha_nms <- character(0)
  mha_exprs <- character(0)

  # Q[q, h, d_k] = sum_c input_q[q, c] * W_q(c, d_k, h) + b_q(d_k, h)
  q_nm <- array(NA_character_, dim = c(T_q, num_heads, key_dim))
  for (q in seq_len(T_q)) {
    for (h in seq_len(num_heads)) {
      for (d in seq_len(key_dim)) {
        terms <- vapply(
          seq_len(C_in),
          function(c) {
            paste0(q_at(q, c), " * ", format_numeric(wq_at(c, d, h)))
          },
          character(1L)
        )
        nm <- paste0("orbital_mha_", lname, "_Q_q", q, "_h", h, "_d", d)
        mha_nms <- c(mha_nms, nm)
        mha_exprs <- c(
          mha_exprs,
          paste0(
            "(",
            paste(terms, collapse = " + "),
            " + ",
            bq_at(d, h),
            ")"
          )
        )
        q_nm[q, h, d] <- nm
      }
    }
  }

  # K[k, h, d_k] = sum_c input_kv[k, c] * W_k(c, d_k, h) + b_k(d_k, h)
  k_nm <- array(NA_character_, dim = c(T_kv, num_heads, key_dim))
  for (k in seq_len(T_kv)) {
    for (h in seq_len(num_heads)) {
      for (d in seq_len(key_dim)) {
        terms <- vapply(
          seq_len(C_in),
          function(c) {
            paste0(kv_at(k, c), " * ", format_numeric(wk_at(c, d, h)))
          },
          character(1L)
        )
        nm <- paste0("orbital_mha_", lname, "_K_k", k, "_h", h, "_d", d)
        mha_nms <- c(mha_nms, nm)
        mha_exprs <- c(
          mha_exprs,
          paste0(
            "(",
            paste(terms, collapse = " + "),
            " + ",
            bk_at(d, h),
            ")"
          )
        )
        k_nm[k, h, d] <- nm
      }
    }
  }

  # V[k, h, d_v] = sum_c input_kv[k, c] * W_v(c, d_v, h) + b_v(d_v, h)
  v_nm <- array(NA_character_, dim = c(T_kv, num_heads, value_dim))
  for (k in seq_len(T_kv)) {
    for (h in seq_len(num_heads)) {
      for (d_v in seq_len(value_dim)) {
        terms <- vapply(
          seq_len(C_in),
          function(c) {
            paste0(kv_at(k, c), " * ", format_numeric(wv_at(c, d_v, h)))
          },
          character(1L)
        )
        nm <- paste0("orbital_mha_", lname, "_V_k", k, "_h", h, "_dv", d_v)
        mha_nms <- c(mha_nms, nm)
        mha_exprs <- c(
          mha_exprs,
          paste0(
            "(",
            paste(terms, collapse = " + "),
            " + ",
            bv_at(d_v, h),
            ")"
          )
        )
        v_nm[k, h, d_v] <- nm
      }
    }
  }

  # Scaled attention scores and softmax (max-stabilised to prevent exp() overflow).
  # For each (q, h) pair: subtract max_{k'} score before exp(), consistent with
  # the standalone Softmax branches in this file.
  scale <- 1.0 / sqrt(as.numeric(key_dim))
  attn_nm <- array(NA_character_, dim = c(T_q, T_kv, num_heads))
  for (q in seq_len(T_q)) {
    for (h in seq_len(num_heads)) {
      # Collect raw scaled score expressions for all key positions
      score_exprs <- character(T_kv)
      for (k in seq_len(T_kv)) {
        dot_terms <- vapply(
          seq_len(key_dim),
          function(d) {
            paste0(backtick(q_nm[q, h, d]), " * ", backtick(k_nm[k, h, d]))
          },
          character(1L)
        )
        score_exprs[k] <- paste0(
          "((",
          paste(dot_terms, collapse = " + "),
          ") * ",
          format_numeric(scale),
          ")"
        )
      }

      # Max-stabilisation: compute row-wise max of scores for (q, h)
      max_nm <- paste0("orbital_mha_", lname, "_smax_q", q, "_h", h)
      mha_nms <- c(mha_nms, max_nm)
      mha_exprs <- c(
        mha_exprs,
        paste0(
          "do.call(pmax, list(",
          paste(score_exprs, collapse = ", "),
          "))"
        )
      )

      exp_nms <- character(T_kv)
      for (k in seq_len(T_kv)) {
        exp_nm <- paste0(
          "orbital_mha_",
          lname,
          "_exp_q",
          q,
          "_k",
          k,
          "_h",
          h
        )
        mha_nms <- c(mha_nms, exp_nm)
        mha_exprs <- c(
          mha_exprs,
          paste0("exp(", score_exprs[k], " - `", max_nm, "`)")
        )
        exp_nms[k] <- exp_nm
      }
      sumexp_nm <- paste0("orbital_mha_", lname, "_sumexp_q", q, "_h", h)
      mha_nms <- c(mha_nms, sumexp_nm)
      mha_exprs <- c(
        mha_exprs,
        paste0("(", paste(backtick(exp_nms), collapse = " + "), ")")
      )
      for (k in seq_len(T_kv)) {
        a_nm <- paste0(
          "orbital_mha_",
          lname,
          "_attn_q",
          q,
          "_k",
          k,
          "_h",
          h
        )
        mha_nms <- c(mha_nms, a_nm)
        mha_exprs <- c(
          mha_exprs,
          paste0("(", backtick(exp_nms[k]), " / ", backtick(sumexp_nm), ")")
        )
        attn_nm[q, k, h] <- a_nm
      }
    }
  }

  # Head outputs: head[q, h, d_v] = sum_k attn[q,k,h] * V[k,h,d_v]
  hout_nm <- array(NA_character_, dim = c(T_q, num_heads, value_dim))
  for (q in seq_len(T_q)) {
    for (h in seq_len(num_heads)) {
      for (d_v in seq_len(value_dim)) {
        terms <- vapply(
          seq_len(T_kv),
          function(k) {
            paste0(
              backtick(attn_nm[q, k, h]),
              " * ",
              backtick(v_nm[k, h, d_v])
            )
          },
          character(1L)
        )
        nm <- paste0(
          "orbital_mha_",
          lname,
          "_hout_q",
          q,
          "_h",
          h,
          "_dv",
          d_v
        )
        mha_nms <- c(mha_nms, nm)
        mha_exprs <- c(
          mha_exprs,
          paste0("(", paste(terms, collapse = " + "), ")")
        )
        hout_nm[q, h, d_v] <- nm
      }
    }
  }

  # Output projection: out[q, d_out] = sum_h sum_dv hout[q,h,dv] * W_o(h,dv,d_out) + b_o
  out_nms <- character(0)
  for (q in seq_len(T_q)) {
    for (d_out in seq_len(C_out_w)) {
      terms <- c()
      for (h in seq_len(num_heads)) {
        for (d_v in seq_len(value_dim)) {
          terms <- c(
            terms,
            paste0(
              backtick(hout_nm[q, h, d_v]),
              " * ",
              format_numeric(wo_at(h, d_v, d_out))
            )
          )
        }
      }
      nm <- paste0("orbital_mha_", lname, "_out_q", q, "_d", d_out)
      mha_nms <- c(mha_nms, nm)
      mha_exprs <- c(
        mha_exprs,
        paste0(
          "(",
          paste(terms, collapse = " + "),
          " + ",
          bo_at(d_out),
          ")"
        )
      )
      out_nms <- c(out_nms, nm)
    }
  }

  state$all_exprs[[lname]] <- stats::setNames(mha_exprs, mha_nms)
  assign(lname, out_nms, envir = expr_reg)
  invisible(NULL)
}

