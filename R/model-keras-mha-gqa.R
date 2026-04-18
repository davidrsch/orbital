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
  .check_attention_mask(lname, topo_map, "GroupedQueryAttention")
  q_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  kv_exprs <- get(
    inbound[min(2L, length(inbound))],
    envir = expr_reg,
    inherits = FALSE
  )

  cfg <- .k3_safe_get_config(l, lname)
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

  shapes <- .mha_extract_sequence_shapes(l, q_exprs, kv_exprs)
  C_in <- shapes$C_in
  T_q <- shapes$T_q
  T_kv <- shapes$T_kv
  C_out <- shapes$C_out

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

  wo_resolved <- .mha_resolve_output_kernel(
    wo,
    num_heads,
    head_dim,
    C_out,
    lname,
    layer_abbr = "GQA"
  )
  wo_at <- wo_resolved$wo_at
  C_out_w <- wo_resolved$C_out_w
  bo_at <- .mha_output_bias_at(wo)

  q_at <- .mha_input_accessor(q_exprs, C_in)
  kv_at <- .mha_input_accessor(kv_exprs, C_in)

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
