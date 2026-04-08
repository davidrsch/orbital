# Shared masking-detection helper used by LSTM and GRU handlers.
# Walk topo_map backwards from lname to detect whether any upstream layer
# is a Keras Masking layer or an Embedding with mask_zero = TRUE.
# Returns TRUE if masking is active, FALSE otherwise.
.detect_masking_upstream <- function(lname, topo_map) {
  visited <- character(0L)
  to_visit <- topo_map[[lname]] %||% character(0L)
  while (length(to_visit) > 0L) {
    nm <- to_visit[[1L]]
    to_visit <- to_visit[-1L]
    if (nm %in% visited) {
      next
    }
    visited <- c(visited, nm)
    if (grepl("^masking", nm, ignore.case = TRUE)) {
      return(TRUE)
    }
    to_visit <- c(to_visit, topo_map[[nm]] %||% character(0L))
  }
  FALSE
}

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

  cfg_l <- tryCatch(l$get_config(), error = function(e) list())
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


.k3_simplernn <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
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


.k3_timedistributed <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
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
      paste0(
        "Keras TimeDistributed layer {.val {lname}} wraps unsupported inner ",
        "layer type {.cls {inner_cls}}."
      ),
      "i" = "orbital currently supports TimeDistributed(Dense) only."
    ))
  }
  invisible(NULL)
}
