# Handler function for the dot-product Attention layer (Luong / scaled dot-product).
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

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
    # Fallback: input spec unavailable; attempt sqrt inference.
    sq <- sqrt(length(q_exprs))
    if (abs(sq - round(sq)) > .Machine$double.eps^0.5) {
      cli::cli_abort(
        c(
          "Attention {.val {lname}}: cannot determine query depth D from model input spec.",
          "x" = "{.code sqrt({length(q_exprs)})} = {sq} is not an integer.",
          "i" = "Ensure the layer is built with an explicit input shape in a Keras Functional model."
        )
      )
    }
    as.integer(round(sq))
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
  n_all <- T_q * (3L * T_k + 2L + D_v)
  all_vals <- character(n_all)
  all_nms <- character(n_all)
  ak <- 1L
  att_out_exprs <- character(T_q * D_v)
  att_out_nms <- character(T_q * D_v)
  ao_k <- 1L

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
      all_vals[[ak]] <- sc_expr_q[k_j]
      all_nms[[ak]] <- sc_names[k_j]
      ak <- ak + 1L
    }

    max_nm <- paste0("orbital_att_", lname, "_max_q", q_i)
    max_expr <- paste0(
      "pmax(",
      paste(backtick(sc_names), collapse = ", "),
      ")"
    )
    all_vals[[ak]] <- max_expr
    all_nms[[ak]] <- max_nm
    ak <- ak + 1L

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
      function(s) {
        paste0("exp(", backtick(s), " - ", backtick(max_nm), ")")
      },
      character(1L)
    )
    for (k_j in seq_len(T_k)) {
      all_vals[[ak]] <- exp_exprs[k_j]
      all_nms[[ak]] <- exp_nms[k_j]
      ak <- ak + 1L
    }

    sum_nm <- paste0("orbital_att_", lname, "_sum_q", q_i)
    sum_expr <- paste0("(", paste(backtick(exp_nms), collapse = " + "), ")")
    all_vals[[ak]] <- sum_expr
    all_nms[[ak]] <- sum_nm
    ak <- ak + 1L

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
      all_vals[[ak]] <- attn_exprs[k_j]
      all_nms[[ak]] <- attn_nms[k_j]
      ak <- ak + 1L
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
      att_out_exprs[[ao_k]] <- out_expr
      att_out_nms[[ao_k]] <- out_nm
      ao_k <- ao_k + 1L
      all_vals[[ak]] <- out_expr
      all_nms[[ak]] <- out_nm
      ak <- ak + 1L
    }
  }
  state$all_exprs[[lname]] <- stats::setNames(all_vals, all_nms)
  assign(lname, att_out_nms, envir = expr_reg)
  invisible(NULL)
}
