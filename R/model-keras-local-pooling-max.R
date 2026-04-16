.k3_adaptivemaxpooling1d <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # AdaptiveMaxPooling1D: adaptive-window max.
  # Same window formula as AdaptiveAveragePooling1D, max instead of mean.
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  n_f <- length(in_exprs)
  o_size <- tryCatch(
    as.integer(l$get_config()$output_size),
    error = function(e) NA_integer_
  )
  if (is.na(o_size) || o_size < 1L) {
    cli::cli_abort(
      "AdaptiveMaxPooling1D: invalid output_size {.val {o_size}}."
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
      "AdaptiveMaxPooling1D: output_size ({o_size}) must not exceed input length ({T_in})."
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
      nm <- paste0("orbital_adaptivemaxpool1d_", lname, "_", idx)
      pool_exprs[[idx]] <- paste0(
        "do.call(pmax, list(",
        paste(cols_bt, collapse = ", "),
        "))"
      )
      pool_nms[[idx]] <- nm
      idx <- idx + 1L
    }
  }
  state$all_exprs[[lname]] <- stats::setNames(pool_exprs, pool_nms)
  assign(lname, pool_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_maxpooling1d <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # MaxPooling1D: sliding-window max across time steps.
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
    "orbital_maxpool1d_",
    lname,
    "_",
    seq_along(window_info$windows)
  )
  pool_exprs <- vapply(
    window_info$windows,
    function(window_bt) {
      if (length(window_bt) > 0L) {
        paste0(
          "do.call(pmax, list(",
          paste(window_bt, collapse = ", "),
          "))"
        )
      } else {
        "-Inf"
      }
    },
    character(1L)
  )
  state$all_exprs[[lname]] <- stats::setNames(pool_exprs, pool_nms)
  assign(lname, pool_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_zeropadding1d <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # ZeroPadding1D: emit literal 0 columns for padded timesteps;
  # pass through interior columns unchanged.
  # Input layout: [T_in × C] flat vector (time-step major).
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

  padding_raw <- tryCatch(l$get_config()$padding, error = function(e) 1L)
  if (is.list(padding_raw)) {
    padding_raw <- as.integer(unlist(padding_raw))
  } else {
    padding_raw <- as.integer(padding_raw)
  }
  pad_left <- if (length(padding_raw) >= 1L) padding_raw[[1L]] else 1L
  pad_right <- if (length(padding_raw) >= 2L) {
    padding_raw[[2L]]
  } else {
    padding_raw[[1L]]
  }

  # Determine C (channels per timestep) from input shape; fallback to 1.
  in_shape <- tryCatch(
    as.integer(unlist(l$input_shape)),
    error = function(e) NULL
  )
  C_feat <- if (!is.null(in_shape) && length(in_shape) >= 1L) {
    tail(in_shape[!is.na(in_shape)], 1L)
  } else {
    1L
  }
  T_in <- as.integer(length(in_exprs) / C_feat)
  T_out <- T_in + pad_left + pad_right

  zero_col <- "0"
  pad_block <- rep(zero_col, C_feat)

  out_exprs <- character(T_out * C_feat)
  out_nms <- character(T_out * C_feat)
  for (tp in seq_len(T_out)) {
    step_nms <- paste0(
      "orbital_zpad_",
      lname,
      "_t",
      tp,
      "_c",
      seq_len(C_feat)
    )
    if (tp <= pad_left || tp > pad_left + T_in) {
      step_exprs <- pad_block
    } else {
      orig_t <- tp - pad_left
      step_exprs <- in_exprs[
        ((orig_t - 1L) * C_feat + 1L):(orig_t * C_feat)
      ]
    }
    idx_s <- (tp - 1L) * C_feat + 1L
    idx_e <- tp * C_feat
    out_exprs[idx_s:idx_e] <- step_exprs
    out_nms[idx_s:idx_e] <- step_nms
  }
  state$all_exprs[[lname]] <- stats::setNames(out_exprs, out_nms)
  assign(lname, out_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_cropping1d <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Cropping1D: remove timesteps from the beginning and end of the sequence.
  # cfg$cropping = [left, right] (number of timesteps to remove from each end).
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

  cfg <- tryCatch(l$get_config(), error = function(e) list())
  cropping_raw <- tryCatch(
    as.integer(unlist(cfg$cropping)),
    error = function(e) c(1L, 1L)
  )
  crop_left <- if (length(cropping_raw) >= 1L) cropping_raw[[1L]] else 1L
  crop_right <- if (length(cropping_raw) >= 2L) {
    cropping_raw[[2L]]
  } else {
    cropping_raw[[1L]]
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
  T_in <- as.integer(length(in_exprs) / C_feat)
  T_out <- T_in - crop_left - crop_right
  if (T_out <= 0L) {
    cli::cli_abort(c(
      "Keras Cropping1D layer {.val {lname}}: cropping removes all {T_in} timesteps.",
      "i" = "cropping = ({crop_left}, {crop_right})"
    ))
  }

  start_idx <- crop_left * C_feat + 1L
  end_idx <- (T_in - crop_right) * C_feat
  out_exprs <- in_exprs[start_idx:end_idx]
  out_nms <- paste0("orbital_crop_", lname, "_h", seq_along(out_exprs))
  state$all_exprs[[lname]] <- stats::setNames(out_exprs, out_nms)
  assign(lname, out_nms, envir = expr_reg)
  invisible(NULL)
}
