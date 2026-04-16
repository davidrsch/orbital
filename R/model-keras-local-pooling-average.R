# Handler functions for local/adaptive 1-D pooling and spatial layers.
# (AdaptiveAveragePooling1D, AveragePooling1D, AdaptiveMaxPooling1D,
#  MaxPooling1D, ZeroPadding1D, Cropping1D).
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

.k3_sliding_pool_windows <- function(
  in_exprs,
  input_shape,
  pool_size,
  stride,
  padding
) {
  C_feat <- if (!is.null(input_shape) && length(input_shape) >= 1L) {
    tail(input_shape[!is.na(input_shape)], 1L)
  } else {
    1L
  }
  T_in <- as.integer(length(in_exprs) / C_feat)
  if (padding == "valid") {
    T_out <- (T_in - pool_size) %/% stride + 1L
    pad_l <- 0L
  } else {
    T_out <- as.integer(ceiling(T_in / stride))
    total_pad <- max(0L, (T_out - 1L) * stride + pool_size - T_in)
    pad_l <- total_pad %/% 2L
  }

  windows <- vector("list", T_out * C_feat)
  idx <- 1L
  for (p in seq_len(T_out) - 1L) {
    for (c in seq_len(C_feat)) {
      kk_seq <- seq_len(pool_size) - 1L
      t_positions <- p * stride + kk_seq - pad_l
      valid_mask <- t_positions >= 0L & t_positions < T_in
      windows[[idx]] <- vapply(
        t_positions[valid_mask],
        function(t) backtick(in_exprs[[t * C_feat + c]]),
        character(1L)
      )
      idx <- idx + 1L
    }
  }

  list(windows = windows, C_feat = C_feat, T_out = T_out)
}

.k3_adaptiveaveragepooling1d <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # AdaptiveAveragePooling1D: adaptive-window mean.
  # For output size O and input length T_in, output position i (0-indexed):
  #   start = floor(i * T_in / O), end = ceiling((i+1) * T_in / O)
  #   output = mean over that window.
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  n_f <- length(in_exprs)
  o_size <- tryCatch(
    as.integer(l$get_config()$output_size),
    error = function(e) NA_integer_
  )
  if (is.na(o_size) || o_size < 1L) {
    cli::cli_abort(
      "AdaptiveAveragePooling1D: invalid output_size {.val {o_size}}."
    )
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
  if (T_in < o_size) {
    cli::cli_abort(
      "AdaptiveAveragePooling1D: output_size ({o_size}) must not exceed input length ({T_in})."
    )
  }
  pool_nms <- character(0L)
  pool_exprs <- character(0L)
  idx <- 1L
  for (i in seq_len(o_size) - 1L) {
    start_t <- as.integer(floor(i * T_in / o_size))
    end_t <- as.integer(ceiling((i + 1L) * T_in / o_size))
    for (c in seq_len(C_feat)) {
      cols_bt <- vapply(
        seq(start_t, end_t - 1L),
        function(t) backtick(in_exprs[[t * C_feat + c]]),
        character(1L)
      )
      n_win <- length(cols_bt)
      nm <- paste0("orbital_adaptiveavgpool1d_", lname, "_", idx)
      pool_exprs[[idx]] <- paste0(
        "(",
        paste(cols_bt, collapse = " + "),
        ") / ",
        n_win
      )
      pool_nms[[idx]] <- nm
      idx <- idx + 1L
    }
  }
  state$all_exprs[[lname]] <- stats::setNames(pool_exprs, pool_nms)
  assign(lname, pool_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_averagepooling1d <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # AveragePooling1D: sliding-window mean across time steps.
  # Input layout: time-step major (T_in × C_feat) — in_exprs[(t-1)*C_feat + c]
  # for 1-indexed timestep t and 1-indexed channel c.
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  n_f <- length(in_exprs)
  pool_size <- tryCatch(
    as.integer(l$get_config()$pool_size),
    error = function(e) NA_integer_
  )
  if (is.na(pool_size)) {
    pool_size <- n_f
  }
  stride <- tryCatch(
    {
      s <- as.integer(unlist(l$get_config()$strides))[[1L]]
      if (!is.na(s)) s else pool_size
    },
    error = function(e) pool_size
  )
  padding <- tryCatch(
    tolower(as.character(l$get_config()$padding)),
    error = function(e) "valid"
  )
  in_shape <- tryCatch(
    as.integer(unlist(l$input_shape)),
    error = function(e) NULL
  )
  window_info <- .k3_sliding_pool_windows(
    in_exprs,
    in_shape,
    pool_size,
    stride,
    padding
  )
  pool_nms <- paste0(
    "orbital_avgpool1d_",
    lname,
    "_",
    seq_along(window_info$windows)
  )
  pool_exprs <- vapply(
    window_info$windows,
    function(window_bt) {
      n_valid <- length(window_bt)
      denom <- if (padding == "same") pool_size else n_valid
      if (n_valid > 0L) {
        paste0("(", paste(window_bt, collapse = " + "), ") / ", denom)
      } else {
        "0"
      }
    },
    character(1L)
  )
  state$all_exprs[[lname]] <- stats::setNames(pool_exprs, pool_nms)
  assign(lname, pool_nms, envir = expr_reg)
  invisible(NULL)
}
