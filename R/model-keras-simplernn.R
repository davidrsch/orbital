# SimpleRNN layer handler for the orbital Keras DAG backend.

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
  cfg_l <- .k3_safe_get_config(l, lname)
  .check_stateful_and_masking(cfg_l, lname, topo_map, "SimpleRNN")
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
