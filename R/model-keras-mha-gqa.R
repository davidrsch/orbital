.k3_groupedqueryattention <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
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

  wts_list <- .mha_extract_sublayer_weights(l, lname, use_bias)
  wq <- wts_list$wq
  wk <- wts_list$wk
  wv <- wts_list$wv
  wo <- wts_list$wo

  if (is.null(wq) || is.null(wk) || is.null(wv) || is.null(wo)) {
    cli::cli_abort(c(
      "GroupedQueryAttention {.val {lname}}: could not extract sub-layer weights.",
      "i" = "Ensure keras3 >= 3.0 with reticulate access to _query_dense etc."
    ))
  }

  wq_at <- .mha_resolve_3d_kernel(
    wq$kernel,
    head_dim,
    num_heads,
    C_in,
    lname,
    "Q",
    layer_abbr = "GQA",
    ambiguity_hint = "Use a configuration where the projected head dimension differs from the head/group count to avoid this ambiguity."
  )
  wk_at <- .mha_resolve_3d_kernel(
    wk$kernel,
    head_dim,
    num_kv,
    C_in,
    lname,
    "K",
    layer_abbr = "GQA",
    ambiguity_hint = "Use a configuration where the projected head dimension differs from the head/group count to avoid this ambiguity."
  )
  wv_at <- .mha_resolve_3d_kernel(
    wv$kernel,
    head_dim,
    num_kv,
    C_in,
    lname,
    "V",
    layer_abbr = "GQA",
    ambiguity_hint = "Use a configuration where the projected head dimension differs from the head/group count to avoid this ambiguity."
  )
  bq_at <- .mha_bias_accessor2(wq$bias, head_dim, num_heads)
  bk_at <- .mha_bias_accessor2(wk$bias, head_dim, num_kv)
  bv_at <- .mha_bias_accessor2(wv$bias, head_dim, num_kv)

  wo_d <- dim(wo$kernel)
  if (length(wo_d) == 3L && wo_d[1L] == num_heads && wo_d[2L] == head_dim) {
    wo_at <- function(h, dv, do_) wo$kernel[h, dv, do_]
    C_out_w <- wo_d[3L]
  } else if (length(wo_d) == 3L && wo_d[1L] == C_out && wo_d[2L] == num_heads) {
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

  # Q / K / V projections via shared helper (model-keras-mha-gqa-projections.R)
  q_proj <- .gqa_linear_project(
    T_q,
    num_heads,
    head_dim,
    C_in,
    q_at,
    wq_at,
    bq_at,
    lname,
    "Q"
  )
  k_proj <- .gqa_linear_project(
    T_kv,
    num_kv,
    head_dim,
    C_in,
    kv_at,
    wk_at,
    bk_at,
    lname,
    "K"
  )
  v_proj <- .gqa_linear_project(
    T_kv,
    num_kv,
    head_dim,
    C_in,
    kv_at,
    wv_at,
    bv_at,
    lname,
    "V"
  )

  q_nm <- q_proj$nm_arr
  k_nm <- k_proj$nm_arr
  v_nm <- v_proj$nm_arr
  gqa_nms <- c(q_proj$nms, k_proj$nms, v_proj$nms)
  gqa_exprs <- c(q_proj$exprs, k_proj$exprs, v_proj$exprs)

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
