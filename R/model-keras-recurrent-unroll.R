# Shared RNN unroll kernel for LSTM, GRU, SimpleRNN.
# Gate variable names: a_input, a_forget, a_cell, a_output.
# Called by .k3_lstm(), .k3_gru(), .k3_simplernn() in model-keras-recurrent.R.

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
        paste0("(", a_forget, " * ", backtick(C_prev), " + ", a_input, " * ", a_cell, ")")
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


