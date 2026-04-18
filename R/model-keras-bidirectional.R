# Bidirectional RNN layer handler for the orbital Keras DAG backend.

.k3_bidirectional <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
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

  cfg_l <- .k3_safe_get_config(l, lname)
  merge_mode <- tryCatch(
    tolower(as.character(cfg_l$merge_mode)),
    error = function(e) "concat"
  )
  if (
    length(merge_mode) == 0L ||
      is.na(merge_mode) ||
      merge_mode == "null" ||
      !nzchar(merge_mode)
  ) {
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
        switch(merge_mode,
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
