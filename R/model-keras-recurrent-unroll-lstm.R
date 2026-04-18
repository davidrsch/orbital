# LSTM unroll kernel: emits per-time-step expressions for gates, cell state,
# and hidden state. Called by .unroll_rnn() when the inner layer is an LSTM.

.unroll_lstm <- function(
  ul_in_exprs,
  ul_wts,
  ul_pfx,
  ul_gate_act,
  ul_cell_act,
  ul_return_seq
) {
  ul_kernel <- ul_wts[[1L]] # (I, 4H)
  ul_rkernel <- ul_wts[[2L]] # (H, 4H)
  ul_H <- as.integer(ncol(ul_kernel) / 4L)
  ul_I <- nrow(ul_kernel)
  ul_T <- as.integer(length(ul_in_exprs) / ul_I)
  ul_bias <- if (length(ul_wts) >= 3L) {
    .flatten_rnn_bias(ul_wts[[3L]])
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
    a_input <- vapply(
      pg[[1L]],
      function(e) activation_expr(ul_gate_act, e),
      character(1L)
    )
    a_forget <- vapply(
      pg[[2L]],
      function(e) activation_expr(ul_gate_act, e),
      character(1L)
    )
    a_cell <- vapply(
      pg[[3L]],
      function(e) activation_expr(ul_cell_act, e),
      character(1L)
    )
    a_output <- vapply(
      pg[[4L]],
      function(e) activation_expr(ul_gate_act, e),
      character(1L)
    )
    C_nms <- paste0(ul_pfx, "_C_t", t, "_h", seq_len(ul_H))
    C_exprs_t <- if (is.null(C_prev)) {
      paste0("(", a_input, " * ", a_cell, ")")
    } else {
      paste0(
        "(",
        a_forget,
        " * ",
        backtick(C_prev),
        " + ",
        a_input,
        " * ",
        a_cell,
        ")"
      )
    }
    H_nms <- paste0(ul_pfx, "_H_t", t, "_h", seq_len(ul_H))
    H_exprs_t <- paste0("(", a_output, " * tanh(", backtick(C_nms), "))")
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
}
