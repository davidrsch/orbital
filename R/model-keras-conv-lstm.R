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
  #
  # LIMITATION (spatial_dim = 1 only):
  #   orbital only supports ConvLSTM1D when the spatial dimension S = 1.
  #   With S = 1 and padding = "same", the spatial convolution at each
  #   timestep reduces to a single-position dot product (using the centre
  #   kernel row), making ConvLSTM1D equivalent to a plain LSTM after weight
  #   slicing.  Input shapes with S > 1 are not supported; orbital will raise
  #   a cli_abort if padding != "same" or strides != 1, which are the only
  #   settings that keep S constant at 1.
  #
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

  cfg_l <- .k3_safe_get_config(l, lname)
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
