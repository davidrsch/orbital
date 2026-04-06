# Handler functions for convolutional layers
# (Conv1DTranspose, DepthwiseConv1D, SeparableConv1D, ConvLSTM1D, Conv1D).
# Called by orbital_keras_dag_impl() in model-keras.R.

.k3_conv1dtranspose <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Conv1DTranspose ─ transposed (fractionally-strided) 1-D convolution.
  # Keras weight layout (same as Conv1D):
  #   kernel : (kernel_size, in_channels, filters)  i.e. (kW, C_in, C_out)
  #   bias   : (filters,)  [optional]
  # Dilations > 1 are not supported (raise cli_abort).
  # Padding: "valid" or "same".
  # W_out formula:
  #   valid: W_out = (W_in - 1) * stride + kW
  #   same:  W_out = W_in * stride
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

  wts <- l$get_weights()
  kern <- wts[[1L]] # (kW, C_in, C_out)
  k_w <- dim(kern)[1L]
  c_in <- dim(kern)[2L]
  c_out <- dim(kern)[3L]
  bias_v <- if (length(wts) >= 2L) as.numeric(wts[[2L]]) else numeric(c_out)

  cfg_l <- tryCatch(l$get_config(), error = function(e) list())
  stride <- tryCatch(as.integer(cfg_l$strides[[1L]]), error = function(e) {
    1L
  })
  dilation <- tryCatch(
    as.integer(cfg_l$dilation_rate[[1L]]),
    error = function(e) 1L
  )
  pad_type <- tryCatch(
    tolower(as.character(cfg_l$padding)),
    error = function(e) "valid"
  )
  if (is.null(stride) || is.na(stride)) {
    stride <- 1L
  }
  if (is.null(dilation) || is.na(dilation)) {
    dilation <- 1L
  }
  if (!nzchar(pad_type %||% "")) {
    pad_type <- "valid"
  }

  if (dilation > 1L) {
    cli::cli_abort(
      "Keras Conv1DTranspose layer {.val {lname}}: dilation_rate > 1 is not supported."
    )
  }

  n_total <- length(in_exprs)
  w_in <- as.integer(n_total / c_in)

  if (pad_type == "same") {
    w_out <- w_in * stride
    pad_total <- max(0L, k_w - stride)
    pad_l <- as.integer(floor(pad_total / 2L))
  } else {
    # "valid"
    w_out <- (w_in - 1L) * stride + k_w
    pad_l <- 0L
  }

  tconv_nms <- character(0L)
  tconv_exprs <- character(0L)

  # For each output position q (0-indexed) and output channel f,
  # gather all input (p, k, c) triples that contribute:
  #   out[q, f] = sum_{p,k,c} in[p, c] * kern[k, c, f]
  # where p * stride + k - pad_l == q  (dilation=1)
  for (q in seq_len(w_out)) {
    q0 <- q - 1L
    for (f in seq_len(c_out)) {
      terms <- character(0L)
      for (k in seq_len(k_w)) {
        k0 <- k - 1L
        # p * stride = q0 + pad_l - k0
        num <- q0 + pad_l - k0
        if (num >= 0L && num %% stride == 0L) {
          p0 <- as.integer(num / stride)
          if (p0 >= 0L && p0 < w_in) {
            for (c in seq_len(c_in)) {
              feat_nm <- in_exprs[p0 * c_in + c]
              wt_val <- format_numeric(kern[k, c, f])
              terms <- c(
                terms,
                paste0("(", backtick(feat_nm), " * ", wt_val, ")")
              )
            }
          }
        }
      }
      b_str <- format_numeric(bias_v[f])
      expr_str <- if (length(terms) == 0L) {
        b_str
      } else {
        paste0("(", paste(c(terms, b_str), collapse = " + "), ")")
      }
      nm <- paste0("orbital_tconv_", lname, "_q", q, "_f", f)
      tconv_nms <- c(tconv_nms, nm)
      tconv_exprs <- c(tconv_exprs, expr_str)
    }
  }
  # Apply inline activation if configured.
  activation_config <- tryCatch(cfg_l$activation, error = function(e) NULL)
  activation <- if (is.null(activation_config)) {
    "linear"
  } else if (is.list(activation_config)) {
    tolower(as.character(activation_config$class_name)[1L])
  } else {
    tolower(as.character(activation_config)[1L])
  }
  if (!nzchar(activation %||% "")) {
    activation <- "linear"
  }
  if (activation != "linear") {
    act_alpha <- tryCatch(
      as.numeric(
        activation_config$config$alpha %||%
          activation_config$config$negative_slope %||%
          1.0
      ),
      error = function(e) 1.0
    )
    tconv_exprs <- vapply(
      tconv_exprs,
      function(e) activation_expr(activation, e, alpha = act_alpha),
      character(1L)
    )
  }
  state$all_exprs[[lname]] <- stats::setNames(tconv_exprs, tconv_nms)
  assign(lname, tconv_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_depthwiseconv1d <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # DepthwiseConv1D ─ channel-wise 1-D convolution (no cross-channel mixing).
  # Keras weight layout:
  #   depthwise_kernel : (kernel_size, in_channels, depth_multiplier)
  #   bias             : (in_channels * depth_multiplier,)  [optional]
  # orbital output convention: T_out × (C_in * depth_mult) columns, time-step major.
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

  wts <- l$get_weights()
  dw_kern <- wts[[1L]] # (kW, C_in, depth_mult)
  k_w <- dim(dw_kern)[1L]
  c_in <- dim(dw_kern)[2L]
  depth_mult <- dim(dw_kern)[3L]
  c_out <- c_in * depth_mult
  bias_v <- if (length(wts) >= 2L) as.numeric(wts[[2L]]) else numeric(c_out)

  cfg_l <- tryCatch(l$get_config(), error = function(e) list())
  stride <- tryCatch(as.integer(cfg_l$strides[[1L]]), error = function(e) {
    1L
  })
  dilation <- tryCatch(
    as.integer(cfg_l$dilation_rate[[1L]]),
    error = function(e) 1L
  )
  pad_type <- tryCatch(
    tolower(as.character(cfg_l$padding)),
    error = function(e) "valid"
  )
  if (is.null(stride) || is.na(stride)) {
    stride <- 1L
  }
  if (is.null(dilation) || is.na(dilation)) {
    dilation <- 1L
  }
  if (!nzchar(pad_type %||% "")) {
    pad_type <- "valid"
  }

  n_total <- length(in_exprs)
  w_in <- as.integer(n_total / c_in)
  k_eff <- dilation * (k_w - 1L) + 1L

  if (pad_type == "same") {
    w_out <- as.integer(ceiling(w_in / stride))
    pad_total <- max(0L, (w_out - 1L) * stride + k_eff - w_in)
    pad_l <- as.integer(floor(pad_total / 2L))
  } else {
    w_out <- as.integer(floor((w_in - k_eff) / stride) + 1L)
    pad_l <- 0L
  }

  dw_nms <- character(0L)
  dw_exprs <- character(0L)
  for (p in seq_len(w_out)) {
    p0 <- p - 1L
    for (c in seq_len(c_in)) {
      for (d in seq_len(depth_mult)) {
        terms <- character(0L)
        for (k in seq_len(k_w)) {
          k0 <- k - 1L
          w_pos <- p0 * stride + k0 * dilation - pad_l
          if (w_pos >= 0L && w_pos < w_in) {
            feat_nm <- in_exprs[w_pos * c_in + c]
            wt_val <- format_numeric(dw_kern[k, c, d])
            terms <- c(
              terms,
              paste0("(", backtick(feat_nm), " * ", wt_val, ")")
            )
          }
        }
        ch_out <- (c - 1L) * depth_mult + d
        b_str <- format_numeric(bias_v[ch_out])
        expr_str <- if (length(terms) == 0L) {
          b_str
        } else {
          paste0("(", paste(c(terms, b_str), collapse = " + "), ")")
        }
        nm <- paste0("orbital_dw_", lname, "_p", p, "_c", ch_out)
        dw_nms <- c(dw_nms, nm)
        dw_exprs <- c(dw_exprs, expr_str)
      }
    }
  }
  # Apply inline activation if configured.
  activation_config <- tryCatch(cfg_l$activation, error = function(e) NULL)
  activation <- if (is.null(activation_config)) {
    "linear"
  } else if (is.list(activation_config)) {
    tolower(as.character(activation_config$class_name)[1L])
  } else {
    tolower(as.character(activation_config)[1L])
  }
  if (!nzchar(activation %||% "")) {
    activation <- "linear"
  }
  if (activation != "linear") {
    act_alpha <- tryCatch(
      as.numeric(
        activation_config$config$alpha %||%
          activation_config$config$negative_slope %||%
          1.0
      ),
      error = function(e) 1.0
    )
    dw_exprs <- vapply(
      dw_exprs,
      function(e) activation_expr(activation, e, alpha = act_alpha),
      character(1L)
    )
  }
  state$all_exprs[[lname]] <- stats::setNames(dw_exprs, dw_nms)
  assign(lname, dw_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_separableconv1d <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # SeparableConv1D ─ depthwise + pointwise 1-D convolution.
  # Keras weight layout:
  #   depthwise_kernel  : (kernel_size, in_channels, depth_multiplier)
  #   pointwise_kernel  : (1, in_channels * depth_multiplier, out_channels)
  #   bias              : (out_channels,)  [optional]
  # Two-stage process:
  #   1. Depthwise conv: C_in * depth_mult intermediate channels per timestep.
  #   2. Pointwise conv: 1x1 linear projection to C_out.
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

  wts <- l$get_weights()
  dw_kern <- wts[[1L]] # (kW, C_in, depth_mult)
  pw_kern <- wts[[2L]] # (1, C_in * depth_mult, C_out)
  k_w <- dim(dw_kern)[1L]
  c_in <- dim(dw_kern)[2L]
  depth_mult <- dim(dw_kern)[3L]
  c_mid <- c_in * depth_mult
  c_out <- dim(pw_kern)[3L]
  bias_v <- if (length(wts) >= 3L) as.numeric(wts[[3L]]) else numeric(c_out)

  cfg_l <- tryCatch(l$get_config(), error = function(e) list())
  stride <- tryCatch(as.integer(cfg_l$strides[[1L]]), error = function(e) {
    1L
  })
  dilation <- tryCatch(
    as.integer(cfg_l$dilation_rate[[1L]]),
    error = function(e) 1L
  )
  pad_type <- tryCatch(
    tolower(as.character(cfg_l$padding)),
    error = function(e) "valid"
  )
  if (is.null(stride) || is.na(stride)) {
    stride <- 1L
  }
  if (is.null(dilation) || is.na(dilation)) {
    dilation <- 1L
  }
  if (!nzchar(pad_type %||% "")) {
    pad_type <- "valid"
  }

  n_total <- length(in_exprs)
  w_in <- as.integer(n_total / c_in)
  k_eff <- dilation * (k_w - 1L) + 1L

  if (pad_type == "same") {
    w_out <- as.integer(ceiling(w_in / stride))
    pad_total <- max(0L, (w_out - 1L) * stride + k_eff - w_in)
    pad_l <- as.integer(floor(pad_total / 2L))
  } else {
    w_out <- as.integer(floor((w_in - k_eff) / stride) + 1L)
    pad_l <- 0L
  }

  # Stage 1: depthwise intermediates (no cross-channel mixing).
  dw_nms <- character(0L)
  dw_exprs <- character(0L)
  for (p in seq_len(w_out)) {
    p0 <- p - 1L
    for (c in seq_len(c_in)) {
      for (d in seq_len(depth_mult)) {
        terms <- character(0L)
        for (k in seq_len(k_w)) {
          k0 <- k - 1L
          w_pos <- p0 * stride + k0 * dilation - pad_l
          if (w_pos >= 0L && w_pos < w_in) {
            feat_nm <- in_exprs[w_pos * c_in + c]
            wt_val <- format_numeric(dw_kern[k, c, d])
            terms <- c(
              terms,
              paste0("(", backtick(feat_nm), " * ", wt_val, ")")
            )
          }
        }
        c2 <- (c - 1L) * depth_mult + d
        expr_str <- if (length(terms) == 0L) {
          "0"
        } else {
          paste0("(", paste(terms, collapse = " + "), ")")
        }
        nm <- paste0("orbital_sep_dw_", lname, "_p", p, "_c", c2)
        dw_nms <- c(dw_nms, nm)
        dw_exprs <- c(dw_exprs, expr_str)
      }
    }
  }

  # Stage 2: pointwise 1x1 projection to C_out.
  sep_nms <- character(0L)
  sep_exprs <- character(0L)
  for (p in seq_len(w_out)) {
    for (f in seq_len(c_out)) {
      terms <- character(0L)
      for (c2 in seq_len(c_mid)) {
        dw_nm <- paste0("orbital_sep_dw_", lname, "_p", p, "_c", c2)
        wt_val <- format_numeric(pw_kern[1L, c2, f])
        terms <- c(
          terms,
          paste0("(", backtick(dw_nm), " * ", wt_val, ")")
        )
      }
      b_str <- format_numeric(bias_v[f])
      expr_str <- if (length(terms) == 0L) {
        b_str
      } else {
        paste0("(", paste(c(terms, b_str), collapse = " + "), ")")
      }
      nm <- paste0("orbital_sep_", lname, "_p", p, "_f", f)
      sep_nms <- c(sep_nms, nm)
      sep_exprs <- c(sep_exprs, expr_str)
    }
  }
  # Apply inline activation if configured (acts on the final pointwise outputs).
  activation_config <- tryCatch(cfg_l$activation, error = function(e) NULL)
  activation <- if (is.null(activation_config)) {
    "linear"
  } else if (is.list(activation_config)) {
    tolower(as.character(activation_config$class_name)[1L])
  } else {
    tolower(as.character(activation_config)[1L])
  }
  if (!nzchar(activation %||% "")) {
    activation <- "linear"
  }
  if (activation != "linear") {
    act_alpha <- tryCatch(
      as.numeric(
        activation_config$config$alpha %||%
          activation_config$config$negative_slope %||%
          1.0
      ),
      error = function(e) 1.0
    )
    sep_exprs <- vapply(
      sep_exprs,
      function(e) activation_expr(activation, e, alpha = act_alpha),
      character(1L)
    )
  }
  state$all_exprs[[lname]] <- c(
    stats::setNames(dw_exprs, dw_nms),
    stats::setNames(sep_exprs, sep_nms)
  )
  assign(lname, sep_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_convlstm1d <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # ConvLSTM1D ─ unrolled for fixed-length sequences with spatial dim = 1.
  # Keras weight layout (IFCO gate order):
  #   kernel           : (kernel_size, in_channels, 4 * filters)
  #   recurrent_kernel : (kernel_size, filters,     4 * filters)
  #   bias             : (4 * filters,)  [optional]
  # Restriction: only spatial dimension S = 1, strides = 1, padding = "same".
  # With S=1 and padding="same", each spatial convolution reduces to a
  # single-position dot product using the centre kernel row (index pad_l).
  # This makes ConvLSTM1D equivalent to a plain LSTM after weight slicing.
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  wts <- l$get_weights() # kernel, recurrent_kernel [, bias]
  kernel <- wts[[1L]] # (k, C, 4F)
  rkernel <- wts[[2L]] # (k, F, 4F)
  k_size <- dim(kernel)[1L]
  C_in <- dim(kernel)[2L]
  F_filt <- as.integer(dim(kernel)[3L] / 4L)
  bias_v <- if (length(wts) >= 3L) {
    as.numeric(wts[[3L]])
  } else {
    numeric(4L * F_filt)
  }

  cfg_l <- tryCatch(l$get_config(), error = function(e) list())
  pad_type <- tryCatch(
    tolower(as.character(cfg_l$padding)),
    error = function(e) "same"
  )
  strides <- tryCatch(as.integer(cfg_l$strides[[1L]]), error = function(e) {
    1L
  })
  if (is.na(strides)) {
    strides <- 1L
  }
  if (!nzchar(pad_type %||% "")) {
    pad_type <- "same"
  }
  if (pad_type != "same" || strides != 1L) {
    cli::cli_abort(c(
      "ConvLSTM1D layer {.val {lname}}: orbital only supports padding='same' and strides=1.",
      "i" = "Got padding={.val {pad_type}}, strides={.val {strides}}."
    ))
  }
  if (isTRUE(tryCatch(as.logical(cfg_l$stateful), error = function(e) FALSE))) {
    cli::cli_abort(c(
      "ConvLSTM1D layer {.val {lname}}: stateful = TRUE is not supported by orbital.",
      "i" = "Only stateless ConvLSTM1Ds can be unrolled."
    ))
  }

  # Infer spatial dimension S from input_shape.
  # Keras input_shape for ConvLSTM1D: (batch, T, S, C_in).
  in_shape <- tryCatch(
    as.integer(unlist(l$input_shape)),
    error = function(e) NULL
  )
  in_shape_valid <- if (!is.null(in_shape)) {
    in_shape[!is.na(in_shape)]
  } else {
    integer(0L)
  }
  # in_shape_valid (after dropping NA batch dim): [T, S, C_in]
  if (length(in_shape_valid) >= 2L) {
    S_spatial <- as.integer(in_shape_valid[length(in_shape_valid) - 1L])
  } else {
    S_spatial <- NA_integer_
  }
  if (is.na(S_spatial) || S_spatial != 1L) {
    cli::cli_abort(c(
      "ConvLSTM1D layer {.val {lname}}: orbital only supports spatial dimension S = 1.",
      "i" = "Got S = {.val {S_spatial}}."
    ))
  }
  # With S=1: in_exprs has T_len * C_in features (the single spatial position is trivial).
  T_len <- as.integer(length(in_exprs) / C_in)

  gate_act <- tryCatch(
    tolower(as.character(cfg_l$recurrent_activation %||% "sigmoid")),
    error = function(e) "sigmoid"
  )
  cell_act <- tryCatch(
    tolower(as.character(cfg_l$activation %||% "tanh")),
    error = function(e) "tanh"
  )
  return_seq <- isTRUE(
    tryCatch(as.logical(cfg_l$return_sequences), error = function(e) FALSE)
  )

  # Centre kernel row (0-indexed pad_l → 1-indexed pad_l + 1L in R).
  pad_l <- as.integer((k_size - 1L) %/% 2L)
  # eff_kern  : (C_in, 4*F_filt) — equivalent to LSTM kernel  (I, 4H)
  # eff_rkernel: (F_filt, 4*F_filt) — equivalent to LSTM recurrent_kernel (H, 4H)
  eff_kern <- kernel[pad_l + 1L, , ]
  dim(eff_kern) <- c(C_in, 4L * F_filt)
  eff_rkernel <- rkernel[pad_l + 1L, , ]
  dim(eff_rkernel) <- c(F_filt, 4L * F_filt)

  # IFCO gate offsets (0-indexed): I=0, F_gate=F_filt, C_tilde=2F_filt, O=3F_filt
  g_offs <- c(0L, F_filt, 2L * F_filt, 3L * F_filt)
  W_gates <- lapply(seq_along(g_offs), function(g) {
    t(eff_kern[, (g_offs[g] + 1L):(g_offs[g] + F_filt)])
  })
  R_gates <- lapply(seq_along(g_offs), function(g) {
    t(eff_rkernel[, (g_offs[g] + 1L):(g_offs[g] + F_filt)])
  })
  B_gates <- lapply(seq_along(g_offs), function(g) {
    as.numeric(bias_v[(g_offs[g] + 1L):(g_offs[g] + F_filt)])
  })

  all_clstm_nms <- character(0L)
  all_clstm_exprs <- character(0L)
  all_H_nms <- list()

  H_prev_nms <- NULL # NULL = zero initial hidden state
  C_prev_nms <- NULL # NULL = zero initial cell state

  for (t in seq_len(T_len)) {
    x_t_exprs <- in_exprs[((t - 1L) * C_in + 1L):(t * C_in)]

    pre_gates <- lapply(seq_along(g_offs), function(g) {
      inp <- build_mlp_pre_act(W_gates[[g]], B_gates[[g]], x_t_exprs)
      if (is.null(H_prev_nms)) {
        inp
      } else {
        rec <- build_mlp_pre_act(R_gates[[g]], numeric(F_filt), H_prev_nms)
        paste0("(", inp, " + ", rec, ")")
      }
    })

    act_I <- vapply(
      pre_gates[[1L]],
      function(e) activation_expr(gate_act, e),
      character(1L)
    )
    act_F <- vapply(
      pre_gates[[2L]],
      function(e) activation_expr(gate_act, e),
      character(1L)
    )
    act_C <- vapply(
      pre_gates[[3L]],
      function(e) activation_expr(cell_act, e),
      character(1L)
    )
    act_O <- vapply(
      pre_gates[[4L]],
      function(e) activation_expr(gate_act, e),
      character(1L)
    )

    C_cur_nms <- paste0(
      "orbital_convlstm_",
      lname,
      "_C_t",
      t,
      "_h",
      seq_len(F_filt)
    )
    C_cur_exprs <- if (is.null(C_prev_nms)) {
      paste0("(", act_I, " * ", act_C, ")")
    } else {
      paste0(
        "(",
        act_F,
        " * ",
        backtick(C_prev_nms),
        " + ",
        act_I,
        " * ",
        act_C,
        ")"
      )
    }

    H_cur_nms <- paste0(
      "orbital_convlstm_",
      lname,
      "_H_t",
      t,
      "_h",
      seq_len(F_filt)
    )
    H_cur_exprs <- paste0("(", act_O, " * tanh(", backtick(C_cur_nms), "))")

    all_clstm_nms <- c(all_clstm_nms, C_cur_nms, H_cur_nms)
    all_clstm_exprs <- c(all_clstm_exprs, C_cur_exprs, H_cur_exprs)
    all_H_nms[[t]] <- H_cur_nms

    H_prev_nms <- H_cur_nms
    C_prev_nms <- C_cur_nms
  }

  state$all_exprs[[lname]] <- stats::setNames(all_clstm_exprs, all_clstm_nms)
  out_nms <- if (return_seq) {
    unlist(all_H_nms, use.names = FALSE)
  } else {
    H_prev_nms
  }
  assign(lname, out_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_conv1d <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Conv1D ─ sliding-window 1-D convolution.
  # Keras weight layout:
  #   kernel  : (kernel_size, in_channels, filters)
  #   bias    : (filters,)  [optional]
  # orbital input convention : T_in × C_in flat columns, time-step major.
  #   in_exprs[(t-1)*C_in + c] = orbital feature for timestep t, channel c (1-indexed).
  #   This matches Keras 3's (batch, T_in, C_in) row-major flattening.
  # orbital output convention: T_out × C_out flat columns, time-step major.
  #   out_names[(p-1)*C_out + f] = filter f at output timestep p (1-indexed).
  #   This matches Keras 3's (batch, T_out, C_out) row-major flattening.
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

  wts <- l$get_weights()
  kern <- wts[[1L]] # (kW, C_in, C_out)
  k_w <- dim(kern)[1L]
  c_in <- dim(kern)[2L]
  c_out <- dim(kern)[3L]
  bias_v <- if (length(wts) >= 2L) as.numeric(wts[[2L]]) else numeric(c_out)

  cfg_l <- tryCatch(l$get_config(), error = function(e) list())
  stride <- tryCatch(as.integer(cfg_l$strides[[1L]]), error = function(e) {
    1L
  })
  dilation <- tryCatch(
    as.integer(cfg_l$dilation_rate[[1L]]),
    error = function(e) 1L
  )
  pad_type <- tryCatch(
    tolower(as.character(cfg_l$padding)),
    error = function(e) "valid"
  )
  if (is.null(stride) || is.na(stride)) {
    stride <- 1L
  }
  if (is.null(dilation) || is.na(dilation)) {
    dilation <- 1L
  }
  if (!nzchar(pad_type %||% "")) {
    pad_type <- "valid"
  }

  n_total <- length(in_exprs)
  w_in <- as.integer(n_total / c_in)
  k_eff <- dilation * (k_w - 1L) + 1L

  if (pad_type == "same") {
    w_out <- as.integer(ceiling(w_in / stride))
    pad_total <- max(0L, (w_out - 1L) * stride + k_eff - w_in)
    pad_l <- as.integer(floor(pad_total / 2L))
  } else {
    w_out <- as.integer(floor((w_in - k_eff) / stride) + 1L)
    pad_l <- 0L
  }

  conv_nms <- character(0L)
  conv_exprs <- character(0L)
  for (p in seq_len(w_out)) {
    p0 <- p - 1L
    for (f in seq_len(c_out)) {
      terms <- character(0L)
      for (c in seq_len(c_in)) {
        for (k in seq_len(k_w)) {
          k0 <- k - 1L
          w_pos <- p0 * stride + k0 * dilation - pad_l
          if (w_pos >= 0L && w_pos < w_in) {
            feat_nm <- in_exprs[w_pos * c_in + c]
            wt_val <- format_numeric(kern[k, c, f])
            terms <- c(
              terms,
              paste0("(", backtick(feat_nm), " * ", wt_val, ")")
            )
          }
        }
      }
      b_str <- format_numeric(bias_v[f])
      expr_str <- if (length(terms) == 0L) {
        b_str
      } else {
        paste0("(", paste(c(terms, b_str), collapse = " + "), ")")
      }
      nm <- paste0("orbital_conv_", lname, "_p", p, "_f", f)
      conv_nms <- c(conv_nms, nm)
      conv_exprs <- c(conv_exprs, expr_str)
    }
  }
  # Apply inline activation if configured.
  activation_config <- tryCatch(cfg_l$activation, error = function(e) NULL)
  activation <- if (is.null(activation_config)) {
    "linear"
  } else if (is.list(activation_config)) {
    tolower(as.character(activation_config$class_name)[1L])
  } else {
    tolower(as.character(activation_config)[1L])
  }
  if (!nzchar(activation %||% "")) {
    activation <- "linear"
  }
  if (activation != "linear") {
    act_alpha <- tryCatch(
      as.numeric(
        activation_config$config$alpha %||%
          activation_config$config$negative_slope %||%
          1.0
      ),
      error = function(e) 1.0
    )
    conv_exprs <- vapply(
      conv_exprs,
      function(e) activation_expr(activation, e, alpha = act_alpha),
      character(1L)
    )
  }
  state$all_exprs[[lname]] <- stats::setNames(conv_exprs, conv_nms)
  assign(lname, conv_nms, envir = expr_reg)
  invisible(NULL)
}
