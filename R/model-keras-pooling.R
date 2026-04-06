# Handler functions for pooling, geometry, and embedding layers
# (GlobalAvgPool, GlobalMaxPool, AdaptiveAvgPool1D, AvgPool1D, AdaptiveMaxPool1D,
#  MaxPool1D, UpSampling1D, GlobalSumPool, ZeroPadding1D, Permute, Cropping1D,
#  RepeatVector, Embedding).
# Called by orbital_keras_dag_impl() in model-keras.R.


.k3_globalaveragepool <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # GlobalAveragePooling1D: reduce feature columns to their row-wise mean
  if (grepl("2d|3d", cls)) {
    cli::cli_abort(
      "GlobalAveragePooling2D/3D is not supported by orbital (requires spatial aggregation)."
    )
  }
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  n_f <- length(in_exprs)
  expr_bt <- backtick(in_exprs)
  gap_nm <- paste0("orbital_gap_", lname, "_1")
  gap_expr <- paste0(
    "(",
    paste(expr_bt, collapse = " + "),
    ") / ",
    n_f
  )
  state$all_exprs[[lname]] <- stats::setNames(gap_expr, gap_nm)
  assign(lname, gap_nm, envir = expr_reg)
  invisible(NULL)
}


.k3_globalmaxpool <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # GlobalMaxPooling1D: reduce feature columns to their row-wise max
  if (grepl("2d|3d", cls)) {
    cli::cli_abort(
      "GlobalMaxPooling2D/3D is not supported by orbital (requires spatial aggregation)."
    )
  }
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  expr_bt <- backtick(in_exprs)
  gmp_nm <- paste0("orbital_gmp_", lname, "_1")
  gmp_expr <- paste0(
    "do.call(pmax, list(",
    paste(expr_bt, collapse = ", "),
    "))"
  )
  state$all_exprs[[lname]] <- stats::setNames(gmp_expr, gmp_nm)
  assign(lname, gmp_nm, envir = expr_reg)
  invisible(NULL)
}


.k3_adaptiveaveragepooling1d <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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


.k3_averagepooling1d <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # AveragePooling1D: sliding-window mean across time steps.
  # Input layout: time-step major (T_in × C_feat) — in_exprs[(t-1)*C_feat + c]
  # for 1-indexed timestep t and 1-indexed channel c.
  if (grepl("2d|3d", cls)) {
    cli::cli_abort(
      "AveragePooling2D/3D is not supported by orbital (requires spatial aggregation)."
    )
  }
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
  C_feat <- if (!is.null(in_shape) && length(in_shape) >= 1L) {
    tail(in_shape[!is.na(in_shape)], 1L)
  } else {
    1L
  }
  T_in <- as.integer(n_f / C_feat)
  if (padding == "valid") {
    T_out <- (T_in - pool_size) %/% stride + 1L
    pad_l <- 0L
  } else {
    T_out <- as.integer(ceiling(T_in / stride))
    total_pad <- max(0L, (T_out - 1L) * stride + pool_size - T_in)
    pad_l <- total_pad %/% 2L
  }
  pool_nms <- character(0)
  pool_exprs <- character(0)
  idx <- 1L
  for (p in seq_len(T_out) - 1L) {
    for (c in seq_len(C_feat)) {
      window_bt <- character(0)
      for (kk in seq_len(pool_size) - 1L) {
        t_pos <- p * stride + kk - pad_l
        if (t_pos >= 0L && t_pos < T_in) {
          flat_idx <- t_pos * C_feat + c
          window_bt <- c(window_bt, backtick(in_exprs[[flat_idx]]))
        }
      }
      n_valid <- length(window_bt)
      nm <- paste0("orbital_avgpool1d_", lname, "_", idx)
      # For "same" padding, Keras 3 divides by pool_size (including
      # implicit zero-padded positions), not by n_valid (in-bounds only).
      # For "valid" padding all windows are full so pool_size == n_valid.
      denom <- if (padding == "same") pool_size else n_valid
      pool_exprs[[idx]] <- if (n_valid > 0L) {
        paste0("(", paste(window_bt, collapse = " + "), ") / ", denom)
      } else {
        "0"
      }
      pool_nms[[idx]] <- nm
      idx <- idx + 1L
    }
  }
  state$all_exprs[[lname]] <- stats::setNames(pool_exprs, pool_nms)
  assign(lname, pool_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_adaptivemaxpooling1d <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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


.k3_maxpooling1d <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # MaxPooling1D: sliding-window max across time steps.
  # Input layout: time-step major (T_in × C_feat) — in_exprs[(t-1)*C_feat + c]
  # for 1-indexed timestep t and 1-indexed channel c.
  if (grepl("2d|3d", cls)) {
    cli::cli_abort(
      "MaxPooling2D/3D is not supported by orbital (requires spatial aggregation)."
    )
  }
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
  C_feat <- if (!is.null(in_shape) && length(in_shape) >= 1L) {
    tail(in_shape[!is.na(in_shape)], 1L)
  } else {
    1L
  }
  T_in <- as.integer(n_f / C_feat)
  if (padding == "valid") {
    T_out <- (T_in - pool_size) %/% stride + 1L
    pad_l <- 0L
  } else {
    T_out <- as.integer(ceiling(T_in / stride))
    total_pad <- max(0L, (T_out - 1L) * stride + pool_size - T_in)
    pad_l <- total_pad %/% 2L
  }
  pool_nms <- character(0)
  pool_exprs <- character(0)
  idx <- 1L
  for (p in seq_len(T_out) - 1L) {
    for (c in seq_len(C_feat)) {
      window_bt <- character(0)
      for (kk in seq_len(pool_size) - 1L) {
        t_pos <- p * stride + kk - pad_l
        if (t_pos >= 0L && t_pos < T_in) {
          flat_idx <- t_pos * C_feat + c
          window_bt <- c(window_bt, backtick(in_exprs[[flat_idx]]))
        }
      }
      nm <- paste0("orbital_maxpool1d_", lname, "_", idx)
      pool_exprs[[idx]] <- if (length(window_bt) > 0L) {
        paste0(
          "do.call(pmax, list(",
          paste(window_bt, collapse = ", "),
          "))"
        )
      } else {
        "-Inf"
      }
      pool_nms[[idx]] <- nm
      idx <- idx + 1L
    }
  }
  state$all_exprs[[lname]] <- stats::setNames(pool_exprs, pool_nms)
  assign(lname, pool_nms, envir = expr_reg)
  invisible(NULL)
}


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


.k3_globalsumpooling <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # GlobalSumPooling1D: row-wise sum of all feature columns.
  # NOTE: GlobalSumPooling1D is not a standard Keras 3 layer; it exists
  # only in keras_cv (keras-cv package). This branch is effectively dead
  # code for stock Keras 3 models. If you are using keras_cv, verify that
  # the class name it produces contains "globalsumpooling"; otherwise this
  # branch will never be reached.
  if (grepl("2d|3d", cls)) {
    cli::cli_abort(
      "GlobalSumPooling2D/3D is not supported by orbital (requires spatial aggregation)."
    )
  }
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  expr_bt <- backtick(in_exprs)
  gsp_nm <- paste0("orbital_gsp_", lname, "_1")
  gsp_expr <- paste0(
    "(",
    paste(expr_bt, collapse = " + "),
    ")"
  )
  state$all_exprs[[lname]] <- stats::setNames(gsp_expr, gsp_nm)
  assign(lname, gsp_nm, envir = expr_reg)
  invisible(NULL)
}


.k3_zeropadding1d <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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

  out_exprs <- character(0L)
  out_nms <- character(0L)
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
    out_exprs <- c(out_exprs, step_exprs)
    out_nms <- c(out_nms, step_nms)
  }
  state$all_exprs[[lname]] <- stats::setNames(out_exprs, out_nms)
  assign(lname, out_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_permute <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # Permute: reorder the axes (dimensions) of the input tensor.
  # In the tabular/1D context the input is [T × C] (time-step major flat).
  # cfg$dims is 1-indexed (Keras convention) over the non-batch axes.
  # For a 2-D input the only supported permutations are (1,2) (no-op) and (2,1).
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  in_names <- in_exprs # column-name vector

  cfg <- tryCatch(l$get_config(), error = function(e) list())
  dims <- tryCatch(as.integer(unlist(cfg$dims)), error = function(e) {
    c(1L, 2L)
  })

  if (length(dims) == 2L && all(dims == c(1L, 2L))) {
    # No-op permutation (1,2): pass through unchanged.
    perm_names <- in_names
  } else if (length(dims) == 2L && all(dims == c(2L, 1L))) {
    # Transpose: swap T and C.
    in_shape <- tryCatch(
      as.integer(unlist(l$input_shape)),
      error = function(e) NULL
    )
    C_feat <- if (!is.null(in_shape) && length(in_shape) >= 1L) {
      tail(in_shape[!is.na(in_shape)], 1L)
    } else {
      1L
    }
    T_in <- length(in_names) / C_feat
    # Transposed: iterate over C first then T (column-major → row-major swap)
    perm_names <- character(length(in_names))
    for (c_i in seq_len(C_feat)) {
      for (t_i in seq_len(T_in)) {
        perm_names[[(c_i - 1L) * T_in + t_i]] <- in_names[[
          (t_i - 1L) * C_feat + c_i
        ]]
      }
    }
  } else {
    cli::cli_abort(
      "Keras Permute layer {.val {lname}}: unsupported dims {paste(dims, collapse=',')}. Only (1,2) and (2,1) are supported."
    )
  }
  perm_out_nms <- paste0(
    "orbital_permute_",
    lname,
    "_h",
    seq_along(perm_names)
  )
  state$all_exprs[[lname]] <- stats::setNames(perm_names, perm_out_nms)
  assign(lname, perm_out_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_cropping1d <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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
    cli::cli_abort(
      "Keras Cropping1D layer {.val {lname}}: cropping ({crop_left},{crop_right}) removes all {T_in} timesteps."
    )
  }

  start_idx <- crop_left * C_feat + 1L
  end_idx <- (T_in - crop_right) * C_feat
  out_exprs <- in_exprs[start_idx:end_idx]
  out_nms <- paste0("orbital_crop_", lname, "_h", seq_along(out_exprs))
  state$all_exprs[[lname]] <- stats::setNames(out_exprs, out_nms)
  assign(lname, out_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_repeatvector <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # RepeatVector: replicate the (flat) input feature vector n times.
  # cfg$n = repetition count; output shape = [n × C_in].
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

  cfg <- tryCatch(l$get_config(), error = function(e) list())
  n_rep <- tryCatch(as.integer(cfg[["n"]]), error = function(e) 1L)
  if (is.na(n_rep) || n_rep < 1L) {
    cli::cli_abort(
      "Keras RepeatVector layer {.val {lname}}: n must be a positive integer, got {n_rep}."
    )
  }

  out_exprs <- rep(in_exprs, times = n_rep)
  out_nms <- paste0("orbital_repvec_", lname, "_h", seq_along(out_exprs))
  state$all_exprs[[lname]] <- stats::setNames(out_exprs, out_nms)
  assign(lname, out_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_embedding <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
  # Embedding: integer index lookup into a dense weight matrix.
  # Keras weight layout:
  #   embeddings : (vocab_size, embed_dim)
  # Input:  T flat integer columns (one token index per timestep, 0-indexed).
  # Output: T x embed_dim flat columns (time-step major).
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

  wts <- l$get_weights()
  emb_mat <- wts[[1L]] # (vocab_size, embed_dim)
  vocab_size <- dim(emb_mat)[1L]
  embed_dim <- dim(emb_mat)[2L]

  if (vocab_size > 50000L) {
    cli::cli_abort(c(
      "Keras Embedding layer {.val {lname}}: vocabulary size {vocab_size} exceeds the",
      " maximum supported limit of 50,000.",
      "x" = "Expanding this layer would generate >50,000 CASE WHEN branches per output column,",
      "     which most SQL engines cannot compile or execute.",
      "i" = "Consider replacing the Embedding layer with a pre-computed lookup table and",
      "     joining the index column against it inside the database instead."
    ))
  }

  if (vocab_size > 10000L) {
    cli::cli_warn(c(
      "Keras Embedding layer {.val {lname}}: vocabulary size {vocab_size} is large.",
      "i" = paste(
        "Generated case_when expressions may be very long.",
        "Consider reducing vocabulary size."
      )
    ))
  }

  T_in <- length(in_exprs) # one column per token position
  emb_nms <- character(0L)
  emb_exprs <- character(0L)
  for (t in seq_len(T_in)) {
    in_col <- in_exprs[t]
    for (d in seq_len(embed_dim)) {
      cases <- vapply(
        seq_len(vocab_size),
        function(i) {
          paste0(
            backtick(in_col),
            " == ",
            i - 1L,
            "L ~ ",
            format_numeric(emb_mat[i, d])
          )
        },
        character(1L)
      )
      expr_str <- paste0(
        "dplyr::case_when(",
        paste(cases, collapse = ", "),
        ", TRUE ~ NA_real_)"
      )
      nm <- paste0("orbital_emb_", lname, "_t", t, "_d", d)
      emb_nms <- c(emb_nms, nm)
      emb_exprs <- c(emb_exprs, expr_str)
    }
  }
  state$all_exprs[[lname]] <- stats::setNames(emb_exprs, emb_nms)
  assign(lname, emb_nms, envir = expr_reg)
  invisible(NULL)
}

