# Handler functions for MHA/GQA attention layers
# (GroupedQueryAttention, MultiHeadAttention).
# Called by orbital_keras_dag_impl() in model-keras.R.

# Internal: retrieve kernel and bias from an EinsumDense sub-layer of a Keras
# MHA/GQA layer.  Returns list(wq, wk, wv, wo) where each element is either
# list(kernel, bias) or NULL if the sub-layer cannot be accessed.
.mha_extract_sublayer_weights <- function(l, lname, use_bias) {
  .get_one <- function(attr_name) {
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
  list(
    wq = .get_one("_query_dense"),
    wk = .get_one("_key_dense"),
    wv = .get_one("_value_dense"),
    wo = .get_one("_output_dense")
  )
}

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

  # Causal-mask limitation: use_causal_mask is a call-time argument in Keras 3
  # and is not stored in the layer config.  orbital cannot enforce or detect it,
  # so predictions for models trained with use_causal_mask=TRUE will be incorrect.
  cli::cli_warn(
    c(
      paste0(
        "MultiHeadAttention {.val {lname}}: {.code use_causal_mask} is a ",
        "call-time argument in Keras 3 and is not saved in the layer config."
      ),
      "i" = paste0(
        "If this layer was called with {.code use_causal_mask = TRUE}, ",
        "orbital cannot enforce the causal mask and predictions will be incorrect."
      ),
      "i" = "See {.code NEWS.md} for details."
    )
  )

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
    if (dim2_size == dim3_size) {
      cli::cli_warn(
        paste0(
          "MHA {.val {lname}}: {name} kernel dims 2 and 3 are equal ({dim2_size}); ",
          "layout is ambiguous \u2014 assuming ({C_in}, {dim2_size}, {dim3_size})."
        ),
        .class = "orbital_mha_kernel_ambiguous"
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
  if (length(wo_d) == 3L && wo_d[1L] == num_heads && wo_d[2L] == value_dim) {
    wo_at <- function(h, dv, do_) wo$kernel[h, dv, do_]
    C_out_w <- wo_d[3L]
  } else if (length(wo_d) == 3L && wo_d[1L] == C_out && wo_d[2L] == num_heads) {
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
