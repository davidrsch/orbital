# Handler for the UpSampling1D layer.
# Called by orbital_keras_dag_impl() in model-keras-dag.R.



.k3_upsampling1d <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # UpSampling1D: repeat each time-step `size` times.
  # Input layout: time-step major (T_in × C_feat).
  # Output: T_out = T_in * size timesteps, same C_feat channels.
  if (grepl("2d|3d", cls)) {
    cli::cli_abort(
      "UpSampling2D/3D is not supported by orbital (requires spatial replication)."
    )
  }
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  n_f <- length(in_exprs)
  size <- tryCatch(
    as.integer(l$get_config()$size),
    error = function(e) NA_integer_
  )
  if (is.na(size) || size < 1L) {
    size <- 2L
  }
  in_shape <- tryCatch(
    as.integer(unlist(l$input_shape)),
    error = function(e) NULL
  )
  C_feat <- if (!is.null(in_shape) && length(in_shape) >= 1L) {
    tail(in_shape[!is.na(in_shape)], 1L)
  } else {
    1L
  }
  T_in <- as.integer(n_f / C_feat)
  T_out <- T_in * size
  up_nms <- character(T_out * C_feat)
  up_exprs <- character(T_out * C_feat)
  idx <- 1L
  for (p in seq_len(T_out) - 1L) {
    t_src <- p %/% size
    for (c in seq_len(C_feat)) {
      src_idx <- t_src * C_feat + c
      up_nms[[idx]] <- paste0("orbital_up1d_", lname, "_", idx)
      up_exprs[[idx]] <- backtick(in_exprs[[src_idx]])
      idx <- idx + 1L
    }
  }
  state$all_exprs[[lname]] <- stats::setNames(up_exprs, up_nms)
  assign(lname, up_nms, envir = expr_reg)
  invisible(NULL)
}
