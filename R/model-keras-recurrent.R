# Helper and handler functions for recurrent layers.
# .unroll_rnn() is a shared helper used by bidirectional and bidirectional.
# Handlers: LSTM, GRU, Bidirectional, SimpleRNN, TimeDistributed.
# Called by orbital_keras_dag_impl() in model-keras.R.


.unroll_rnn <- function(ul, ul_in_exprs, ul_pfx) {
  inner_cls <- tolower(class(ul)[1L])
  is_lstm_inner <- grepl("\\blstm\\b", inner_cls, perl = TRUE)
  ul_wts <- ul$get_weights()
  ul_cfg <- tryCatch(ul$get_config(), error = function(e) list())
  ul_return_seq <- isTRUE(
    tryCatch(as.logical(ul_cfg$return_sequences), error = function(e) FALSE)
  )
  ul_gate_act <- tryCatch(
    tolower(as.character(ul_cfg$recurrent_activation %||% "sigmoid")),
    error = function(e) "sigmoid"
  )
  ul_cell_act <- tryCatch(
    tolower(as.character(ul_cfg$activation %||% "tanh")),
    error = function(e) "tanh"
  )

  if (is_lstm_inner) {
    ul_kernel <- ul_wts[[1L]] # (I, 4H)
    ul_rkernel <- ul_wts[[2L]] # (H, 4H)
    ul_H <- as.integer(ncol(ul_kernel) / 4L)
    ul_I <- nrow(ul_kernel)
    ul_T <- as.integer(length(ul_in_exprs) / ul_I)
    ul_bias <- if (length(ul_wts) >= 3L) {
      as.numeric(ul_wts[[3L]])
    } else {
      numeric(4L * ul_H)
    }
    ul_g_offs <- c(0L, ul_H, 2L * ul_H, 3L * ul_H)
    ul_W <- lapply(seq_along(ul_g_offs), function(g) {
      t(ul_kernel[, (ul_g_offs[g] + 1L):(ul_g_offs[g] + ul_H)])
    })
    ul_R <- lapply(seq_along(ul_g_offs), function(g) {
      t(ul_rkernel[, (ul_g_offs[g] + 1L):(ul_g_offs[g] + ul_H)])
    })
    ul_B <- lapply(seq_along(ul_g_offs), function(g) {
      as.numeric(ul_bias[(ul_g_offs[g] + 1L):(ul_g_offs[g] + ul_H)])
    })

    nms <- character(0L)
    exprs <- character(0L)
    all_H_nms <- list()
    H_prev <- NULL
    C_prev <- NULL
    for (t in seq_len(ul_T)) {
      xt <- ul_in_exprs[((t - 1L) * ul_I + 1L):(t * ul_I)]
      pg <- lapply(seq_along(ul_g_offs), function(g) {
        inp <- build_mlp_pre_act(ul_W[[g]], ul_B[[g]], xt)
        if (is.null(H_prev)) {
          inp
        } else {
          paste0(
            "(",
            inp,
            " + ",
            build_mlp_pre_act(ul_R[[g]], numeric(ul_H), H_prev),
            ")"
          )
        }
      })
      aI <- vapply(
        pg[[1L]],
        function(e) activation_expr(ul_gate_act, e),
        character(1L)
      )
      aF <- vapply(
        pg[[2L]],
        function(e) activation_expr(ul_gate_act, e),
        character(1L)
      )
      aC <- vapply(
        pg[[3L]],
        function(e) activation_expr(ul_cell_act, e),
        character(1L)
      )
      aO <- vapply(
        pg[[4L]],
        function(e) activation_expr(ul_gate_act, e),
        character(1L)
      )
      C_nms <- paste0(ul_pfx, "_C_t", t, "_h", seq_len(ul_H))
      C_exprs_t <- if (is.null(C_prev)) {
        paste0("(", aI, " * ", aC, ")")
      } else {
        paste0("(", aF, " * ", backtick(C_prev), " + ", aI, " * ", aC, ")")
      }
      H_nms <- paste0(ul_pfx, "_H_t", t, "_h", seq_len(ul_H))
      H_exprs_t <- paste0("(", aO, " * tanh(", backtick(C_nms), "))")
      nms <- c(nms, C_nms, H_nms)
      exprs <- c(exprs, C_exprs_t, H_exprs_t)
      all_H_nms[[t]] <- H_nms
      H_prev <- H_nms
      C_prev <- C_nms
    }
    out_nms <- if (ul_return_seq) {
      unlist(all_H_nms, use.names = FALSE)
    } else {
      H_prev
    }
    list(
      nms = nms,
      exprs = exprs,
      out_nms = out_nms,
      all_H_nms = all_H_nms,
      T = ul_T,
      H = ul_H
    )
  } else {
    # GRU
    ul_kernel <- ul_wts[[1L]] # (I, 3H)
    ul_rkernel <- ul_wts[[2L]] # (H, 3H)
    ul_H <- as.integer(ncol(ul_kernel) / 3L)
    ul_I <- nrow(ul_kernel)
    ul_T <- as.integer(length(ul_in_exprs) / ul_I)
    if (length(ul_wts) >= 3L) {
      raw_b <- ul_wts[[3L]]
      if (!is.null(dim(raw_b)) && length(dim(raw_b)) == 2L) {
        ul_b_inp <- as.numeric(raw_b[1L, ])
        ul_b_rec <- as.numeric(raw_b[2L, ])
      } else {
        ba <- as.numeric(raw_b)
        ul_b_inp <- ba[seq_len(3L * ul_H)]
        ul_b_rec <- ba[seq_len(3L * ul_H) + 3L * ul_H]
      }
    } else {
      ul_b_inp <- numeric(3L * ul_H)
      ul_b_rec <- numeric(3L * ul_H)
    }
    ul_reset_after <- isTRUE(
      tryCatch(as.logical(ul_cfg$reset_after), error = function(e) TRUE)
    )
    ul_g_offs <- c(0L, ul_H, 2L * ul_H)
    ul_B_comb <- lapply(seq_along(ul_g_offs), function(g) {
      idx <- (ul_g_offs[g] + 1L):(ul_g_offs[g] + ul_H)
      ul_b_inp[idx] + ul_b_rec[idx]
    })
    ul_W <- lapply(seq_along(ul_g_offs), function(g) {
      t(ul_kernel[, (ul_g_offs[g] + 1L):(ul_g_offs[g] + ul_H)])
    })
    ul_R <- lapply(seq_along(ul_g_offs), function(g) {
      t(ul_rkernel[, (ul_g_offs[g] + 1L):(ul_g_offs[g] + ul_H)])
    })

    nms <- character(0L)
    exprs <- character(0L)
    all_H_nms <- list()
    H_prev <- NULL
    for (t in seq_len(ul_T)) {
      xt <- ul_in_exprs[((t - 1L) * ul_I + 1L):(t * ul_I)]
      z_pre <- {
        inp <- build_mlp_pre_act(ul_W[[1L]], ul_B_comb[[1L]], xt)
        if (is.null(H_prev)) {
          inp
        } else {
          paste0(
            "(",
            inp,
            " + ",
            build_mlp_pre_act(ul_R[[1L]], numeric(ul_H), H_prev),
            ")"
          )
        }
      }
      r_pre <- {
        inp <- build_mlp_pre_act(ul_W[[2L]], ul_B_comb[[2L]], xt)
        if (is.null(H_prev)) {
          inp
        } else {
          paste0(
            "(",
            inp,
            " + ",
            build_mlp_pre_act(ul_R[[2L]], numeric(ul_H), H_prev),
            ")"
          )
        }
      }
      z_nms_t <- paste0(ul_pfx, "_z_t", t, "_h", seq_len(ul_H))
      r_nms_t <- paste0(ul_pfx, "_r_t", t, "_h", seq_len(ul_H))
      z_exprs_t <- vapply(
        z_pre,
        function(e) activation_expr(ul_gate_act, e),
        character(1L)
      )
      r_exprs_t <- vapply(
        r_pre,
        function(e) activation_expr(ul_gate_act, e),
        character(1L)
      )
      h_idx <- (ul_g_offs[3L] + 1L):(ul_g_offs[3L] + ul_H)
      h_inp <- build_mlp_pre_act(
        ul_W[[3L]],
        if (ul_reset_after) ul_b_inp[h_idx] else ul_B_comb[[3L]],
        xt
      )
      ul_R_h <- ul_R[[3L]] # (H, H)
      ht_exprs_t <- vapply(
        seq_len(ul_H),
        function(h) {
          if (is.null(H_prev)) {
            activation_expr(ul_cell_act, h_inp[h])
          } else if (ul_reset_after) {
            rec_sum <- paste(
              paste0(backtick(H_prev), " * ", format_numeric(ul_R_h[h, ])),
              collapse = " + "
            )
            rec_wb <- paste0(
              "(",
              rec_sum,
              " + ",
              format_numeric(ul_b_rec[h_idx[h]]),
              ")"
            )
            activation_expr(
              ul_cell_act,
              paste0(
                "(",
                h_inp[h],
                " + ",
                backtick(r_nms_t[h]),
                " * ",
                rec_wb,
                ")"
              )
            )
          } else {
            coupled <- paste(
              paste0(
                backtick(r_nms_t),
                " * ",
                backtick(H_prev),
                " * ",
                format_numeric(ul_R_h[h, ])
              ),
              collapse = " + "
            )
            activation_expr(
              ul_cell_act,
              paste0("(", h_inp[h], " + ", coupled, ")")
            )
          }
        },
        character(1L)
      )
      ht_nms_t <- paste0(ul_pfx, "_ht_t", t, "_h", seq_len(ul_H))
      H_nms <- paste0(ul_pfx, "_H_t", t, "_h", seq_len(ul_H))
      H_cur_exprs <- if (is.null(H_prev)) {
        paste0("((1 - ", backtick(z_nms_t), ") * ", backtick(ht_nms_t), ")")
      } else {
        paste0(
          "((1 - ",
          backtick(z_nms_t),
          ") * ",
          backtick(ht_nms_t),
          " + ",
          backtick(z_nms_t),
          " * ",
          backtick(H_prev),
          ")"
        )
      }
      nms <- c(nms, z_nms_t, r_nms_t, ht_nms_t, H_nms)
      exprs <- c(exprs, z_exprs_t, r_exprs_t, ht_exprs_t, H_cur_exprs)
      all_H_nms[[t]] <- H_nms
      H_prev <- H_nms
    }
    out_nms <- if (ul_return_seq) {
      unlist(all_H_nms, use.names = FALSE)
    } else {
      H_prev
    }
    list(
      nms = nms,
      exprs = exprs,
      out_nms = out_nms,
      all_H_nms = all_H_nms,
      T = ul_T,
      H = ul_H
    )
  }
}


.k3_lstm <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # LSTM ─ unrolled for fixed-length sequences.
  # Keras weight layout (IFCO gate order, columns 1:H = I, H+1:2H = F, ...):
  #   kernel           : (input_size,  4 * units)
  #   recurrent_kernel : (units,        4 * units)
  #   bias             : (4 * units,)  [optional, Keras sums input+recurrent biases]
  # orbital input: T * I flat columns (time-step major).
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  wts <- l$get_weights() # kernel, recurrent_kernel [, bias]

  kernel <- wts[[1L]] # (I, 4H)
  rkernel <- wts[[2L]] # (H, 4H)
  H <- as.integer(ncol(kernel) / 4L)
  I_feat <- nrow(kernel)
  T_len <- as.integer(length(in_exprs) / I_feat)
  bias_v <- if (length(wts) >= 3L) {
    as.numeric(wts[[3L]])
  } else {
    numeric(4L * H)
  }

  # Get activation config (defaults: sigmoid for gates I/F/O, tanh for C)
  cfg_l <- tryCatch(l$get_config(), error = function(e) list())
  if (
    isTRUE(tryCatch(as.logical(cfg_l$stateful), error = function(e) FALSE))
  ) {
    cli::cli_abort(c(
      "LSTM layer {.val {lname}}: stateful = TRUE is not supported by orbital.",
      "i" = "Only stateless LSTMs (stateful = FALSE, the Keras default) can be unrolled into SQL."
    ))
  }
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

  # IFCO gate offsets (0-indexed column starts):
  # I=0, F=H, C=2H, O=3H
  g_offs <- c(0L, H, 2L * H, 3L * H)

  # Transposed weight matrices for gate g (0-indexed g=0..3):
  #   W_gates[[g+1]] : (H × I_feat), used with build_mlp_pre_act
  #   R_gates[[g+1]] : (H × H),      used with build_mlp_pre_act
  W_gates <- lapply(seq_along(g_offs), function(g) {
    t(kernel[, (g_offs[g] + 1L):(g_offs[g] + H)])
  })
  R_gates <- lapply(seq_along(g_offs), function(g) {
    t(rkernel[, (g_offs[g] + 1L):(g_offs[g] + H)])
  })
  B_gates <- lapply(seq_along(g_offs), function(g) {
    as.numeric(bias_v[(g_offs[g] + 1L):(g_offs[g] + H)])
  })

  all_lstm_nms <- character(0L)
  all_lstm_exprs <- character(0L)
  all_H_nms <- list() # collect per-timestep H names for return_sequences

  H_prev_nms <- NULL # NULL = zero initial hidden state
  C_prev_nms <- NULL # NULL = zero initial cell state

  for (t in seq_len(T_len)) {
    x_t_exprs <- in_exprs[((t - 1L) * I_feat + 1L):(t * I_feat)]

    # Pre-activations for all four gates (IFCO order)
    pre_gates <- lapply(seq_along(g_offs), function(g) {
      inp <- build_mlp_pre_act(W_gates[[g]], B_gates[[g]], x_t_exprs)
      if (is.null(H_prev_nms)) {
        inp # H_prev = 0 → recurrent contribution is 0
      } else {
        rec <- build_mlp_pre_act(R_gates[[g]], numeric(H), H_prev_nms)
        paste0("(", inp, " + ", rec, ")")
      }
    })
    # pre_gates[[1]] = I gate, [[2]] = F gate, [[3]] = C gate, [[4]] = O gate

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

    # Cell state C_t = F_t * C_{t-1} + I_t * C̃_t
    C_cur_nms <- paste0("orbital_lstm_", lname, "_C_t", t, "_h", seq_len(H))
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

    # Hidden state H_t = O_t * tanh(C_t)
    H_cur_nms <- paste0("orbital_lstm_", lname, "_H_t", t, "_h", seq_len(H))
    H_cur_exprs <- paste0("(", act_O, " * tanh(", backtick(C_cur_nms), "))")

    all_lstm_nms <- c(all_lstm_nms, C_cur_nms, H_cur_nms)
    all_lstm_exprs <- c(all_lstm_exprs, C_cur_exprs, H_cur_exprs)
    all_H_nms[[t]] <- H_cur_nms

    H_prev_nms <- H_cur_nms
    C_prev_nms <- C_cur_nms
  }

  state$all_exprs[[lname]] <- stats::setNames(all_lstm_exprs, all_lstm_nms)
  out_nms <- if (return_seq) {
    unlist(all_H_nms, use.names = FALSE)
  } else {
    H_prev_nms # last timestep's hidden state
  }
  assign(lname, out_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_gru <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # GRU ─ unrolled for fixed-length sequences.
  # Keras weight layout (ZRH gate order):
  #   kernel           : (input_size, 3 * units)
  #   recurrent_kernel : (units,       3 * units)
  #   bias             : (2, 3 * units)  row 1 = input bias, row 2 = recurrent bias
  # Gate equations (ONNX default, linear_before_reset = 0):
  #   z_t = f(x@Wz + H_prev@Rz + bz)
  #   r_t = f(x@Wr + H_prev@Rr + br)
  #   h̃_t = g(x@Wh + (r_t ⊙ H_prev)@Rh + bh)
  #   H_t = (1 − z_t) ⊙ h̃_t + z_t ⊙ H_prev
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  wts <- l$get_weights() # kernel, recurrent_kernel [, bias]

  kernel <- wts[[1L]] # (I, 3H)
  rkernel <- wts[[2L]] # (H, 3H)
  H <- as.integer(ncol(kernel) / 3L)
  I_feat <- nrow(kernel)
  T_len <- as.integer(length(in_exprs) / I_feat)

  # Bias: (2, 3H) matrix or flat (6H) vector
  if (length(wts) >= 3L) {
    raw_b <- wts[[3L]]
    if (!is.null(dim(raw_b)) && length(dim(raw_b)) == 2L) {
      b_input <- as.numeric(raw_b[1L, ])
      b_recur <- as.numeric(raw_b[2L, ])
    } else {
      ba <- as.numeric(raw_b)
      b_input <- ba[seq_len(3L * H)]
      b_recur <- ba[seq_len(3L * H) + 3L * H]
    }
  } else {
    b_input <- numeric(3L * H)
    b_recur <- numeric(3L * H)
  }

  # Get activation config (defaults: sigmoid for Z/R, tanh for H-tilde)
  cfg_l <- tryCatch(l$get_config(), error = function(e) list())
  if (
    isTRUE(tryCatch(as.logical(cfg_l$stateful), error = function(e) FALSE))
  ) {
    cli::cli_abort(c(
      "GRU layer {.val {lname}}: stateful = TRUE is not supported by orbital.",
      "i" = "Only stateless GRUs (stateful = FALSE, the Keras default) can be unrolled into SQL."
    ))
  }
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
  # When reset_after = TRUE (the Keras 3 default), the recurrent bias for
  # the h-tilde gate is applied INSIDE the reset-gate multiplication:
  #   h̃_t = g(x@Wh + b_hx + r_t[h] * (H_prev@Rh[h] + b_hh[h]))
  # When reset_after = FALSE (legacy), the combined bias goes outside:
  #   h̃_t = g((r ⊙ H_prev)@Rh + x@Wh + b_h)
  reset_after <- isTRUE(
    tryCatch(as.logical(cfg_l$reset_after), error = function(e) TRUE)
  )

  # ZRH gate offsets (0-indexed column starts): Z=0, R=H, H-tilde=2H
  g_offs <- c(0L, H, 2L * H)

  # Combined bias (input + recurrent) per gate
  B_gates <- lapply(seq_along(g_offs), function(g) {
    idx <- (g_offs[g] + 1L):(g_offs[g] + H)
    b_input[idx] + b_recur[idx]
  })

  # Transposed weight matrices: W_gates[[g]][h, i] = weight from input i to unit h
  W_gates <- lapply(seq_along(g_offs), function(g) {
    t(kernel[, (g_offs[g] + 1L):(g_offs[g] + H)])
  })
  # R_gates[[g]][h, hid] = weight from recurrent unit hid to output unit h
  R_gates <- lapply(seq_along(g_offs), function(g) {
    t(rkernel[, (g_offs[g] + 1L):(g_offs[g] + H)])
  })

  all_gru_nms <- character(0L)
  all_gru_exprs <- character(0L)
  all_H_nms <- list()

  H_prev_nms <- NULL # NULL = zero initial hidden state

  for (t in seq_len(T_len)) {
    x_t_exprs <- in_exprs[((t - 1L) * I_feat + 1L):(t * I_feat)]

    # z gate (update gate)
    z_pre <- {
      inp <- build_mlp_pre_act(W_gates[[1L]], B_gates[[1L]], x_t_exprs)
      if (is.null(H_prev_nms)) {
        inp
      } else {
        rec <- build_mlp_pre_act(R_gates[[1L]], numeric(H), H_prev_nms)
        paste0("(", inp, " + ", rec, ")")
      }
    }
    z_nms <- paste0("orbital_gru_", lname, "_z_t", t, "_h", seq_len(H))
    z_exprs <- vapply(
      z_pre,
      function(e) activation_expr(gate_act, e),
      character(1L)
    )

    # r gate (reset gate)
    r_pre <- {
      inp <- build_mlp_pre_act(W_gates[[2L]], B_gates[[2L]], x_t_exprs)
      if (is.null(H_prev_nms)) {
        inp
      } else {
        rec <- build_mlp_pre_act(R_gates[[2L]], numeric(H), H_prev_nms)
        paste0("(", inp, " + ", rec, ")")
      }
    }
    r_nms <- paste0("orbital_gru_", lname, "_r_t", t, "_h", seq_len(H))
    r_exprs <- vapply(
      r_pre,
      function(e) activation_expr(gate_act, e),
      character(1L)
    )

    # h̃ gate
    # reset_after=TRUE  (Keras 3 default): h̃ = g(x@Wh + b_hx + r[h]*(H_prev@Rh + b_hh))
    # reset_after=FALSE (legacy):          h̃ = g((r ⊙ H_prev)@Rh + x@Wh + b_h)
    R_h <- R_gates[[3L]] # (H, H): R_h[h, hid] = recurrent weight
    h_idx <- (g_offs[3L] + 1L):(g_offs[3L] + H)
    h_inp <- build_mlp_pre_act(
      W_gates[[3L]],
      if (reset_after) b_input[h_idx] else B_gates[[3L]],
      x_t_exprs
    )

    h_tilde_exprs <- vapply(
      seq_len(H),
      function(h) {
        if (is.null(H_prev_nms)) {
          # H_prev = 0 → recurrent term vanishes entirely
          activation_expr(cell_act, h_inp[h])
        } else if (reset_after) {
          # Correct reset_after=TRUE formula:
          # r[h] * (\sum_j H_prev[j]*R_h[h,j] + b_hh[h])
          rec_sum <- paste(
            paste0(backtick(H_prev_nms), " * ", format_numeric(R_h[h, ])),
            collapse = " + "
          )
          rec_with_bias <- paste0(
            "(",
            rec_sum,
            " + ",
            format_numeric(b_recur[h_idx[h]]),
            ")"
          )
          coupled <- paste0(backtick(r_nms[h]), " * ", rec_with_bias)
          activation_expr(
            cell_act,
            paste0("(", h_inp[h], " + ", coupled, ")")
          )
        } else {
          # Legacy reset_after=FALSE formula: (r ⊙ H_prev) @ Rh
          coupled <- paste(
            paste0(
              backtick(r_nms),
              " * ",
              backtick(H_prev_nms),
              " * ",
              format_numeric(R_h[h, ])
            ),
            collapse = " + "
          )
          activation_expr(
            cell_act,
            paste0("(", h_inp[h], " + ", coupled, ")")
          )
        }
      },
      character(1L)
    )
    h_tilde_nms <- paste0(
      "orbital_gru_",
      lname,
      "_ht_t",
      t,
      "_h",
      seq_len(H)
    )

    # H_t = (1 - z_t) * h̃_t + z_t * H_prev
    H_cur_nms <- paste0("orbital_gru_", lname, "_H_t", t, "_h", seq_len(H))
    H_cur_exprs <- if (is.null(H_prev_nms)) {
      paste0("((1 - ", backtick(z_nms), ") * ", backtick(h_tilde_nms), ")")
    } else {
      paste0(
        "((1 - ",
        backtick(z_nms),
        ") * ",
        backtick(h_tilde_nms),
        " + ",
        backtick(z_nms),
        " * ",
        backtick(H_prev_nms),
        ")"
      )
    }

    # Order matters for SQL column dependencies: z, r, h̃, H
    all_gru_nms <- c(all_gru_nms, z_nms, r_nms, h_tilde_nms, H_cur_nms)
    all_gru_exprs <- c(
      all_gru_exprs,
      z_exprs,
      r_exprs,
      h_tilde_exprs,
      H_cur_exprs
    )
    all_H_nms[[t]] <- H_cur_nms

    H_prev_nms <- H_cur_nms
  }

  state$all_exprs[[lname]] <- stats::setNames(all_gru_exprs, all_gru_nms)
  out_nms <- if (return_seq) {
    unlist(all_H_nms, use.names = FALSE)
  } else {
    H_prev_nms
  }
  assign(lname, out_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_bidirectional <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # Bidirectional wrapper: runs forward and backward passes of an inner
  # LSTM or GRU and concatenates their outputs.
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

  fwd_layer <- tryCatch(l$forward_layer, error = function(e) NULL)
  bwd_layer <- tryCatch(l$backward_layer, error = function(e) NULL)
  if (is.null(fwd_layer) || is.null(bwd_layer)) {
    cli::cli_abort(
      c(
        "Bidirectional layer {.val {lname}} does not expose forward_layer / backward_layer.",
        "i" = "Ensure you are using Keras 3 (>= 3.0)."
      )
    )
  }
  fwd_cls_inner <- tolower(class(fwd_layer)[1L])
  if (!grepl("\\blstm\\b|\\bgru\\b", fwd_cls_inner, perl = TRUE)) {
    cli::cli_abort(
      "Bidirectional layer {.val {lname}}: inner layer {.cls {fwd_cls_inner}} is not LSTM or GRU."
    )
  }

  cfg_l <- tryCatch(l$get_config(), error = function(e) list())
  merge_mode <- tryCatch(
    tolower(as.character(cfg_l$merge_mode)),
    error = function(e) "concat"
  )
  if (is.na(merge_mode) || merge_mode == "null" || !nzchar(merge_mode)) {
    merge_mode <- "concat"
  }
  if (!merge_mode %in% c("concat", "sum", "mul", "ave")) {
    cli::cli_abort(
      "Bidirectional layer {.val {lname}}: merge_mode = {.val {merge_mode}} is not supported."
    )
  }

  # Infer I_feat and T_len from forward layer kernel shape
  fwd_wts <- fwd_layer$get_weights()
  I_feat_bd <- nrow(fwd_wts[[1L]])
  T_len_bd <- as.integer(length(in_exprs) / I_feat_bd)

  fwd_pfx <- paste0("orbital_bidir_fwd_", lname)
  fwd <- .unroll_rnn(fwd_layer, in_exprs, fwd_pfx)

  rev_exprs <- unlist(
    lapply(rev(seq_len(T_len_bd)), function(t) {
      in_exprs[((t - 1L) * I_feat_bd + 1L):(t * I_feat_bd)]
    }),
    use.names = FALSE
  )
  bwd_pfx <- paste0("orbital_bidir_bwd_", lname)
  bwd <- .unroll_rnn(bwd_layer, rev_exprs, bwd_pfx)

  state$all_exprs[[lname]] <- stats::setNames(
    c(fwd$exprs, bwd$exprs),
    c(fwd$nms, bwd$nms)
  )

  # Determine the inner layer's return_sequences flag
  inner_return_seq <- isTRUE(
    tryCatch(
      as.logical(fwd_layer$get_config()$return_sequences),
      error = function(e) FALSE
    )
  )

  if (merge_mode == "concat") {
    out_nms_bd <- if (inner_return_seq) {
      # Per timestep t: [fwd[t] || bwd[T-t+1]]
      # (bwd processed reversed input; bwd step k = original step T-k+1)
      unlist(
        lapply(seq_len(fwd$T), function(t) {
          c(fwd$all_H_nms[[t]], bwd$all_H_nms[[fwd$T - t + 1L]])
        }),
        use.names = FALSE
      )
    } else {
      c(fwd$out_nms, bwd$out_nms)
    }
  } else {
    # sum / mul / ave: emit merged intermediate expressions
    fwd_final <- if (inner_return_seq) {
      unlist(fwd$all_H_nms, use.names = FALSE)
    } else {
      fwd$out_nms
    }
    bwd_final <- if (inner_return_seq) {
      unlist(
        lapply(seq_len(fwd$T), function(t) {
          bwd$all_H_nms[[fwd$T - t + 1L]]
        }),
        use.names = FALSE
      )
    } else {
      bwd$out_nms
    }
    merge_nms <- paste0(
      "orbital_bidir_merge_",
      lname,
      "_",
      seq_along(fwd_final)
    )
    merge_exprs_bd <- vapply(
      seq_along(fwd_final),
      function(i) {
        switch(
          merge_mode,
          sum = paste0(
            "(",
            backtick(fwd_final[i]),
            " + ",
            backtick(bwd_final[i]),
            ")"
          ),
          mul = paste0(
            "(",
            backtick(fwd_final[i]),
            " * ",
            backtick(bwd_final[i]),
            ")"
          ),
          ave = paste0(
            "((",
            backtick(fwd_final[i]),
            " + ",
            backtick(bwd_final[i]),
            ") / 2)"
          )
        )
      },
      character(1L)
    )
    state$all_exprs[[lname]] <- c(
      state$all_exprs[[lname]],
      stats::setNames(merge_exprs_bd, merge_nms)
    )
    out_nms_bd <- merge_nms
  }
  assign(lname, out_nms_bd, envir = expr_reg)
  invisible(NULL)
}


.k3_simplernn <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # SimpleRNN: h_t = activation(x_t @ W + h_{t-1} @ U + b)
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  wts <- l$get_weights() # kernel (I, H), recurrent_kernel (H, H) [, bias (H,)]
  kernel <- wts[[1L]] # (I, H)
  rkernel <- wts[[2L]] # (H, H)
  H <- ncol(kernel)
  I_feat <- nrow(kernel)
  T_len <- as.integer(length(in_exprs) / I_feat)
  bias_v <- if (length(wts) >= 3L) as.numeric(wts[[3L]]) else numeric(H)
  cfg_l <- tryCatch(l$get_config(), error = function(e) list())
  activation <- tryCatch(
    tolower(as.character(cfg_l$activation %||% "tanh")),
    error = function(e) "tanh"
  )
  return_seq <- isTRUE(
    tryCatch(as.logical(cfg_l$return_sequences), error = function(e) FALSE)
  )
  W_t <- t(kernel) # (H, I)
  U_t <- t(rkernel) # (H, H)

  all_srnn_nms <- character(0L)
  all_srnn_exprs <- character(0L)
  all_H_nms <- list()
  H_prev_nms <- NULL

  for (t in seq_len(T_len)) {
    x_t_exprs <- in_exprs[((t - 1L) * I_feat + 1L):(t * I_feat)]
    pre_act <- build_mlp_pre_act(W_t, bias_v, x_t_exprs)
    if (!is.null(H_prev_nms)) {
      rec <- build_mlp_pre_act(U_t, numeric(H), H_prev_nms)
      pre_act <- paste0("(", pre_act, " + ", rec, ")")
    }
    H_nms <- paste0("orbital_srnn_", lname, "_H_t", t, "_h", seq_len(H))
    H_exprs_t <- vapply(
      pre_act,
      function(e) activation_expr(activation, e),
      character(1L)
    )
    all_srnn_nms <- c(all_srnn_nms, H_nms)
    all_srnn_exprs <- c(all_srnn_exprs, H_exprs_t)
    all_H_nms[[t]] <- H_nms
    H_prev_nms <- H_nms
  }
  state$all_exprs[[lname]] <- stats::setNames(all_srnn_exprs, all_srnn_nms)
  out_nms <- if (return_seq) {
    unlist(all_H_nms, use.names = FALSE)
  } else {
    H_prev_nms
  }
  assign(lname, out_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_timedistributed <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # TimeDistributed: apply a layer independently to each timestep.
  # Supports Dense inner layer; other inner types raise cli_abort.
  # Input:  T_in x C_in flat columns (time-step major).
  # Output: T_in x units flat columns (time-step major).
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

  inner <- tryCatch(l$layer, error = function(e) NULL)
  if (is.null(inner)) {
    cli::cli_abort(
      "Keras TimeDistributed layer {.val {lname}}: cannot access wrapped layer."
    )
  }
  inner_cls <- tolower(class(inner)[1L])

  if (grepl("dense", inner_cls)) {
    inner_wts <- inner$get_weights()
    if (length(inner_wts) < 1L) {
      cli::cli_abort(
        "Keras TimeDistributed(Dense) layer {.val {lname}}: no weights found."
      )
    }
    kern_td <- t(inner_wts[[1L]]) # (units x C_in)
    bias_td <- if (length(inner_wts) >= 2L) {
      as.numeric(inner_wts[[2L]])
    } else {
      numeric(nrow(kern_td))
    }
    units <- nrow(kern_td)
    c_in_td <- ncol(kern_td)
    T_steps <- as.integer(length(in_exprs) / c_in_td)

    activation_config_td <- tryCatch(
      inner$get_config()$activation,
      error = function(e) NULL
    )
    activation_td <- if (is.null(activation_config_td)) {
      "linear"
    } else if (is.list(activation_config_td)) {
      tolower(as.character(activation_config_td$class_name)[1L])
    } else {
      tolower(as.character(activation_config_td)[1L])
    }
    if (!nzchar(activation_td)) {
      activation_td <- "linear"
    }
    act_alpha_td <- if (
      is.list(activation_config_td) &&
        !is.null(activation_config_td[["config"]])
    ) {
      activation_config_td[["config"]][["alpha"]]
    } else {
      NULL
    }

    td_nms <- character(0L)
    td_exprs <- character(0L)
    for (t in seq_len(T_steps)) {
      t_in <- in_exprs[((t - 1L) * c_in_td + 1L):(t * c_in_td)]
      pre_act <- build_mlp_pre_act(kern_td, bias_td, t_in)
      act_str <- vapply(
        pre_act,
        function(z) activation_expr(activation_td, z, alpha = act_alpha_td),
        character(1L)
      )
      step_nms <- paste0(
        "orbital_td_",
        lname,
        "_t",
        t,
        "_h",
        seq_len(units)
      )
      td_nms <- c(td_nms, step_nms)
      td_exprs <- c(td_exprs, act_str)
    }
    state$all_exprs[[lname]] <- stats::setNames(td_exprs, td_nms)
    assign(lname, td_nms, envir = expr_reg)
  } else {
    cli::cli_abort(c(
      "Keras TimeDistributed layer {.val {lname}} wraps unsupported inner layer type {.cls {inner_cls}}.",
      "i" = "orbital currently supports TimeDistributed(Dense) only."
    ))
  }
  invisible(NULL)
}

