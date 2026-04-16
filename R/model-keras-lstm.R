# Handler function for the LSTM layer (unrolled fixed-length LSTM).
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

.k3_lstm <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # LSTM ─ unrolled for fixed-length sequences.
  # Keras weight layout (IFCO gate order, columns 1:H = I, H+1:2H = F, ...):
  #   kernel           : (input_size,  4 * units)
  #   recurrent_kernel : (units,        4 * units)
  #   bias             : (4 * units,) flat OR (2, 4 * units) 2-row matrix
  #                      row 1 = input bias, row 2 = recurrent bias
  # orbital input: T * I flat columns (time-step major).
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  wts <- l$get_weights() # kernel, recurrent_kernel [, bias]

  kernel <- wts[[1L]] # (I, 4H)
  rkernel <- wts[[2L]] # (H, 4H)
  H <- as.integer(ncol(kernel) / 4L)
  I_feat <- nrow(kernel)
  T_len <- as.integer(length(in_exprs) / I_feat)
  # Keras 3 may return (2, 4H) matrix or a flat (4H,) vector.
  # Sum rows if matrix (input bias + recurrent bias), otherwise use as-is.
  bias_v <- if (length(wts) >= 3L) {
    .flatten_rnn_bias(wts[[3L]])
  } else {
    numeric(4L * H)
  }

  # Get activation config (defaults: sigmoid for gates I/F/O, tanh for C)
  cfg_l <- tryCatch(l$get_config(), error = function(e) list())
  .check_stateful_and_masking(cfg_l, lname, topo_map, "LSTM")
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
  go_backwards <- isTRUE(cfg_l$go_backwards)

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

  for (t in if (go_backwards) rev(seq_len(T_len)) else seq_len(T_len)) {
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
    if (go_backwards) {
      unlist(rev(all_H_nms), use.names = FALSE)
    } else {
      unlist(all_H_nms, use.names = FALSE)
    }
  } else {
    H_prev_nms # last timestep's hidden state
  }
  assign(lname, out_nms, envir = expr_reg)
  invisible(NULL)
}
