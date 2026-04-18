# Handler function for the AdditiveAttention (Bahdanau) layer.
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

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
  .check_attention_mask(lname, topo_map, "AdditiveAttention")
  q_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  v_exprs <- get(inbound[2L], envir = expr_reg, inherits = FALSE)
  k_exprs <- if (length(inbound) >= 3L) {
    get(inbound[3L], envir = expr_reg, inherits = FALSE)
  } else {
    v_exprs
  }

  cfg_aa <- .k3_safe_get_config(l, lname)
  use_scale_aa <- tryCatch(
    as.logical(cfg_aa$use_scale),
    error = function(e) FALSE
  )
  aa_scale <- if (isTRUE(use_scale_aa)) {
    wts_aa <- l$get_weights()
    if (length(wts_aa) < 1L) {
      cli::cli_abort(c(
        "Keras AdditiveAttention layer {.val {lname}}: use_scale = TRUE but no scale weight was exported.",
        i = "Expected at least 1 weight tensor; got {length(wts_aa)}.",
        i = "Rebuild / retrain the model so the additive-attention scale is materialised."
      ))
    }
    as.numeric(wts_aa[[1L]])[1L]
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

  n_all_aa <- T_q_aa * (3L * T_k_aa + 2L + D_v_aa)
  all_vals_aa <- character(n_all_aa)
  all_nms_aa <- character(n_all_aa)
  ak_aa <- 1L
  att_out_exprs_aa <- character(T_q_aa * D_v_aa)
  att_out_nms_aa <- character(T_q_aa * D_v_aa)
  ao_k_aa <- 1L

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
      all_vals_aa[[ak_aa]] <- sc_expr_q_aa[k_j]
      all_nms_aa[[ak_aa]] <- sc_names_aa[k_j]
      ak_aa <- ak_aa + 1L
    }
    max_nm_aa <- paste0("orbital_addatt_", lname, "_max_q", q_i)
    max_expr_aa <- paste0(
      "pmax(",
      paste(backtick(sc_names_aa), collapse = ", "),
      ")"
    )
    all_vals_aa[[ak_aa]] <- max_expr_aa
    all_nms_aa[[ak_aa]] <- max_nm_aa
    ak_aa <- ak_aa + 1L
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
      all_vals_aa[[ak_aa]] <- exp_exprs_aa[k_j]
      all_nms_aa[[ak_aa]] <- exp_nms_aa[k_j]
      ak_aa <- ak_aa + 1L
    }
    sum_nm_aa <- paste0("orbital_addatt_", lname, "_sum_q", q_i)
    sum_expr_aa <- paste0(
      "(",
      paste(backtick(exp_nms_aa), collapse = " + "),
      ")"
    )
    all_vals_aa[[ak_aa]] <- sum_expr_aa
    all_nms_aa[[ak_aa]] <- sum_nm_aa
    ak_aa <- ak_aa + 1L
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
      function(e) {
        paste0("(", backtick(e), " / ", backtick(sum_nm_aa), ")")
      },
      character(1L)
    )
    for (k_j in seq_len(T_k_aa)) {
      all_vals_aa[[ak_aa]] <- attn_exprs_aa[k_j]
      all_nms_aa[[ak_aa]] <- attn_nms_aa[k_j]
      ak_aa <- ak_aa + 1L
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
      att_out_exprs_aa[[ao_k_aa]] <- out_expr_aa
      att_out_nms_aa[[ao_k_aa]] <- out_nm_aa
      ao_k_aa <- ao_k_aa + 1L
      all_vals_aa[[ak_aa]] <- out_expr_aa
      all_nms_aa[[ak_aa]] <- out_nm_aa
      ak_aa <- ak_aa + 1L
    }
  }
  state$all_exprs[[lname]] <- stats::setNames(all_vals_aa, all_nms_aa)
  assign(lname, att_out_nms_aa, envir = expr_reg)
  invisible(NULL)
}
