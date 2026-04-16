# Handler function for the GRU layer (unrolled fixed-length GRU).
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

.k3_gru <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
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
      b_recur <- numeric(3L * H)
    }
  } else {
    b_input <- numeric(3L * H)
    b_recur <- numeric(3L * H)
  }

  # Get activation config (defaults: sigmoid for Z/R, tanh for H-tilde)
  cfg_l <- tryCatch(l$get_config(), error = function(e) list())
  .check_stateful_and_masking(cfg_l, lname, topo_map, "GRU")
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

  for (t in if (go_backwards) rev(seq_len(T_len)) else seq_len(T_len)) {
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
            paste0(
              backtick(H_prev_nms),
              " * ",
              format_numeric(R_h[h, ])
            ),
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
      paste0(
        "((1 - ",
        backtick(z_nms),
        ") * ",
        backtick(h_tilde_nms),
        ")"
      )
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
    if (go_backwards) {
      unlist(rev(all_H_nms), use.names = FALSE)
    } else {
      unlist(all_H_nms, use.names = FALSE)
    }
  } else {
    H_prev_nms
  }
  assign(lname, out_nms, envir = expr_reg)
  invisible(NULL)
}
