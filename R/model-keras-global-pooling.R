# Handler functions for global 1-D pooling layers.
# (GlobalAveragePooling1D, GlobalMaxPooling1D, GlobalSumPooling1D).
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

.k3_globalaveragepool <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # GlobalAveragePooling1D: reduce per-channel timestep columns to their mean
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  n_f <- length(in_exprs)
  in_shape <- tryCatch(
    as.integer(unlist(l$input_shape)),
    error = function(e) NULL
  )
  C_feat <- if (!is.null(in_shape) && length(in_shape) >= 1L) {
    tail(in_shape[!is.na(in_shape)], 1L)
  } else {
    1L
  }
  T_len <- as.integer(n_f / C_feat)
  pool_nms <- character(0L)
  pool_exprs <- character(0L)
  for (c in seq_len(C_feat)) {
    cols_bt <- vapply(
      seq_len(T_len),
      function(t) backtick(in_exprs[[(t - 1L) * C_feat + c]]),
      character(1L)
    )
    nm <- paste0("orbital_gap_", lname, "_", c)
    pool_exprs <- c(
      pool_exprs,
      paste0(
        "(",
        paste(cols_bt, collapse = " + "),
        ") / ",
        T_len
      )
    )
    pool_nms <- c(pool_nms, nm)
  }
  state$all_exprs[[lname]] <- stats::setNames(pool_exprs, pool_nms)
  assign(lname, pool_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_globalmaxpool <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # GlobalMaxPooling1D: reduce per-channel timestep columns to their max
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  n_f <- length(in_exprs)
  in_shape <- tryCatch(
    as.integer(unlist(l$input_shape)),
    error = function(e) NULL
  )
  C_feat <- if (!is.null(in_shape) && length(in_shape) >= 1L) {
    tail(in_shape[!is.na(in_shape)], 1L)
  } else {
    1L
  }
  T_len <- as.integer(n_f / C_feat)
  pool_nms <- character(0L)
  pool_exprs <- character(0L)
  for (c in seq_len(C_feat)) {
    cols_bt <- vapply(
      seq_len(T_len),
      function(t) backtick(in_exprs[[(t - 1L) * C_feat + c]]),
      character(1L)
    )
    nm <- paste0("orbital_gmp_", lname, "_", c)
    pool_exprs <- c(
      pool_exprs,
      paste0(
        "do.call(pmax, list(",
        paste(cols_bt, collapse = ", "),
        "))"
      )
    )
    pool_nms <- c(pool_nms, nm)
  }
  state$all_exprs[[lname]] <- stats::setNames(pool_exprs, pool_nms)
  assign(lname, pool_nms, envir = expr_reg)
  invisible(NULL)
}


.k3_globalsumpooling <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # GlobalSumPooling1D: per-channel sum over all timestep columns.
  # NOTE: GlobalSumPooling1D is not a standard Keras 3 layer; it exists
  # only in keras_cv (keras-cv package). This branch is effectively dead
  # code for stock Keras 3 models. If you are using keras_cv, verify that
  # the class name it produces contains "globalsumpooling"; otherwise this
  # branch will never be reached.
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  n_f <- length(in_exprs)
  in_shape <- tryCatch(
    as.integer(unlist(l$input_shape)),
    error = function(e) NULL
  )
  C_feat <- if (!is.null(in_shape) && length(in_shape) >= 1L) {
    tail(in_shape[!is.na(in_shape)], 1L)
  } else {
    1L
  }
  T_len <- as.integer(n_f / C_feat)
  pool_nms <- character(0L)
  pool_exprs <- character(0L)
  for (c in seq_len(C_feat)) {
    cols_bt <- vapply(
      seq_len(T_len),
      function(t) backtick(in_exprs[[(t - 1L) * C_feat + c]]),
      character(1L)
    )
    nm <- paste0("orbital_gsp_", lname, "_", c)
    pool_exprs <- c(
      pool_exprs,
      paste0(
        "(",
        paste(cols_bt, collapse = " + "),
        ")"
      )
    )
    pool_nms <- c(pool_nms, nm)
  }
  state$all_exprs[[lname]] <- stats::setNames(pool_exprs, pool_nms)
  assign(lname, pool_nms, envir = expr_reg)
  invisible(NULL)
}
