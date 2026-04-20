.k3_multiheadattention <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # MultiHeadAttention: scaled dot-product self-attention or cross-attention.
  # Supports: fixed-length sequences; use_causal_mask TRUE/FALSE; use_bias TRUE/FALSE.
  # Input layout: time-step major (T × C_in) flat vector.
  # Weight extraction via reticulate::py_get_attr on internal EinsumDense sub-layers.
  inbound <- topo_map[[lname]]
  .check_attention_mask(lname, topo_map, "MultiHeadAttention")
  q_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  kv_exprs <- get(
    inbound[min(2L, length(inbound))],
    envir = expr_reg,
    inherits = FALSE
  )

  cfg <- .k3_safe_get_config(l, lname)
  num_heads <- as.integer(cfg$num_heads %||% 1L)
  key_dim <- as.integer(cfg$key_dim %||% 1L)
  value_dim <- as.integer(cfg$value_dim %||% key_dim)
  use_bias <- isTRUE(as.logical(cfg$use_bias %||% TRUE))

  shapes <- .mha_extract_sequence_shapes(l, q_exprs, kv_exprs)
  C_in <- shapes$C_in
  T_q <- shapes$T_q
  T_kv <- shapes$T_kv
  C_out <- shapes$C_out

  # Retrieve kernel and bias from an EinsumDense sub-layer (handles _attr naming).
  wts_list <- .mha_extract_sublayer_weights(l, lname, use_bias)
  wq <- wts_list$wq
  wk <- wts_list$wk
  wv <- wts_list$wv
  wo <- wts_list$wo

  if (is.null(wq) || is.null(wk) || is.null(wv) || is.null(wo)) {
    cli::cli_abort(
      "MultiHeadAttention {.val {lname}}: could not extract sub-layer projection weights.",
      "i" = "Ensure keras3 >= 3.0 with reticulate access to _query_dense, _key_dense, etc."
    )
  }

  # Causal mask: with static T_q / T_kv we can materialise the upper-triangular
  # mask at translation time. For each query position q, restrict attention to
  # key positions k <= q (i.e. mask[q, k] = FALSE when k > q, equivalent to
  # adding -inf to the score before softmax, which means exp(-inf)=0 drops the
  # term out of both the softmax numerator and denominator).
  cfg_main <- .k3_safe_get_config(l, lname)
  use_causal_mask <- isTRUE(as.logical(cfg_main$use_causal_mask %||% FALSE))
  if (use_causal_mask) {
    if (!is.finite(T_q) || T_q < 1L || !is.finite(T_kv) || T_kv < 1L) {
      cli::cli_abort(c(
        "MultiHeadAttention layer {.val {lname}}: {.code use_causal_mask = TRUE} requires static sequence lengths.",
        "x" = "Got T_q = {T_q}, T_kv = {T_kv}.",
        "i" = "Rebuild the model with explicit (non-None) sequence lengths on the query and key/value inputs."
      ))
    }
  }
  # Valid key indices per query (1-based). For non-causal: all keys; for
  # causal: keys up to and including the query position.
  causal_keys <- function(q) {
    if (use_causal_mask) seq_len(min(q, T_kv)) else seq_len(T_kv)
  }

  # Resolve the shared Q/K/V kernel layouts via the common helper.
  wq_at <- .mha_resolve_3d_kernel(
    wq$kernel,
    key_dim,
    num_heads,
    C_in,
    lname,
    "Q",
    layer_abbr = "MHA",
    ambiguity_hint = "Use a configuration with {.code key_dim != num_heads} to avoid this ambiguity."
  )
  wk_at <- .mha_resolve_3d_kernel(
    wk$kernel,
    key_dim,
    num_heads,
    C_in,
    lname,
    "K",
    layer_abbr = "MHA",
    ambiguity_hint = "Use a configuration with {.code key_dim != num_heads} to avoid this ambiguity."
  )
  wv_at <- .mha_resolve_3d_kernel(
    wv$kernel,
    value_dim,
    num_heads,
    C_in,
    lname,
    "V",
    layer_abbr = "MHA",
    ambiguity_hint = "Use a configuration with {.code value_dim != num_heads} to avoid this ambiguity."
  )

  bq_at <- .mha_bias_accessor2(wq$bias, key_dim, num_heads)
  bk_at <- .mha_bias_accessor2(wk$bias, key_dim, num_heads)
  bv_at <- .mha_bias_accessor2(wv$bias, value_dim, num_heads)

  # Output projection: (num_heads, value_dim, C_out) or transposed.
  wo_resolved <- .mha_resolve_output_kernel(
    wo,
    num_heads,
    value_dim,
    C_out,
    lname,
    layer_abbr = "MHA"
  )
  wo_at <- wo_resolved$wo_at
  C_out_w <- wo_resolved$C_out_w
  bo_at <- .mha_output_bias_at(wo)

  # Helper: column name for an input at (timestep t, channel c)
  q_at <- .mha_input_accessor(q_exprs, C_in)
  kv_at <- .mha_input_accessor(kv_exprs, C_in)

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
      ks <- causal_keys(q)
      # Collect raw scaled score expressions for valid key positions only.
      score_exprs <- character(length(ks))
      for (ki in seq_along(ks)) {
        k <- ks[ki]
        dot_terms <- vapply(
          seq_len(key_dim),
          function(d) {
            paste0(backtick(q_nm[q, h, d]), " * ", backtick(k_nm[k, h, d]))
          },
          character(1L)
        )
        score_exprs[ki] <- paste0(
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

      exp_nms <- character(length(ks))
      for (ki in seq_along(ks)) {
        k <- ks[ki]
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
          paste0("exp(", score_exprs[ki], " - `", max_nm, "`)")
        )
        exp_nms[ki] <- exp_nm
      }
      sumexp_nm <- paste0("orbital_mha_", lname, "_sumexp_q", q, "_h", h)
      mha_nms <- c(mha_nms, sumexp_nm)
      mha_exprs <- c(
        mha_exprs,
        paste0("(", paste(backtick(exp_nms), collapse = " + "), ")")
      )
      for (ki in seq_along(ks)) {
        k <- ks[ki]
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
          paste0("(", backtick(exp_nms[ki]), " / ", backtick(sumexp_nm), ")")
        )
        attn_nm[q, k, h] <- a_nm
      }
    }
  }

  # Head outputs: head[q, h, d_v] = sum_k attn[q,k,h] * V[k,h,d_v]
  # Under causal masking, only sum over keys k <= q (other terms are 0).
  hout_nm <- array(NA_character_, dim = c(T_q, num_heads, value_dim))
  for (q in seq_len(T_q)) {
    ks <- causal_keys(q)
    for (h in seq_len(num_heads)) {
      for (d_v in seq_len(value_dim)) {
        terms <- vapply(
          ks,
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
