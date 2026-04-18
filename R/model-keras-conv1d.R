# Handler function for Conv1D (standard sliding-window 1-D convolution).
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

.k3_conv1d <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Conv1D ─ sliding-window 1-D convolution.
  # Keras weight layout:
  #   kernel  : (kernel_size, in_channels, filters)
  #   bias    : (filters,)  [optional]
  # orbital input convention : T_in × C_in flat columns, time-step major.
  #   in_exprs[(t-1)*C_in + c] = orbital feature for timestep t, channel c (1-indexed).
  #   This matches Keras 3's (batch, T_in, C_in) row-major flattening.
  # orbital output convention: T_out × C_out flat columns, time-step major.
  #   out_names[(p-1)*C_out + f] = filter f at output timestep p (1-indexed).
  #   This matches Keras 3's (batch, T_out, C_out) row-major flattening.
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

  wts <- .k3_get_weights(l, lname, required = 1L, names = c("kernel", "bias"))
  kern <- wts[[1L]] # (kW, C_in, C_out)
  k_w <- dim(kern)[1L]
  c_in <- dim(kern)[2L]
  c_out <- dim(kern)[3L]
  bias_v <- if (length(wts) >= 2L) as.numeric(wts[[2L]]) else numeric(c_out)

  cfg_l <- .k3_safe_get_config(l, lname)
  stride <- tryCatch(as.integer(cfg_l$strides[[1L]]), error = function(e) {
    1L
  })
  dilation <- tryCatch(
    as.integer(cfg_l$dilation_rate[[1L]]),
    error = function(e) 1L
  )
  pad_type <- tryCatch(
    tolower(as.character(cfg_l$padding)),
    error = function(e) "valid"
  )
  if (is.null(stride) || is.na(stride)) {
    stride <- 1L
  }
  if (is.null(dilation) || is.na(dilation)) {
    dilation <- 1L
  }
  if (!nzchar(pad_type %||% "")) {
    pad_type <- "valid"
  }

  n_total <- length(in_exprs)
  w_in <- as.integer(n_total / c_in)
  k_eff <- dilation * (k_w - 1L) + 1L

  if (pad_type == "same") {
    w_out <- as.integer(ceiling(w_in / stride))
    pad_total <- max(0L, (w_out - 1L) * stride + k_eff - w_in)
    pad_l <- as.integer(floor(pad_total / 2L))
  } else {
    w_out <- as.integer(floor((w_in - k_eff) / stride) + 1L)
    pad_l <- 0L
  }

  conv_nms <- character(0L)
  conv_exprs <- character(0L)
  for (p in seq_len(w_out)) {
    p0 <- p - 1L
    for (f in seq_len(c_out)) {
      terms <- character(0L)
      for (c in seq_len(c_in)) {
        for (k in seq_len(k_w)) {
          k0 <- k - 1L
          w_pos <- p0 * stride + k0 * dilation - pad_l
          if (w_pos >= 0L && w_pos < w_in) {
            feat_nm <- in_exprs[w_pos * c_in + c]
            wt_val <- format_numeric(kern[k, c, f])
            terms <- c(
              terms,
              paste0("(", backtick(feat_nm), " * ", wt_val, ")")
            )
          }
        }
      }
      b_str <- format_numeric(bias_v[f])
      expr_str <- if (length(terms) == 0L) {
        b_str
      } else {
        paste0("(", paste(c(terms, b_str), collapse = " + "), ")")
      }
      nm <- paste0("orbital_conv_", lname, "_p", p, "_f", f)
      conv_nms <- c(conv_nms, nm)
      conv_exprs <- c(conv_exprs, expr_str)
    }
  }
  # Apply inline activation if configured.
  activation_config <- tryCatch(cfg_l$activation, error = function(e) NULL)
  activation <- if (is.null(activation_config)) {
    "linear"
  } else if (is.list(activation_config)) {
    tolower(as.character(activation_config$class_name)[1L])
  } else {
    tolower(as.character(activation_config)[1L])
  }
  if (!nzchar(activation %||% "")) {
    activation <- "linear"
  }
  if (activation != "linear") {
    act_alpha <- tryCatch(
      as.numeric(
        activation_config$config$alpha %||%
          activation_config$config$negative_slope %||%
          1.0
      ),
      error = function(e) 1.0
    )
    conv_exprs <- vapply(
      conv_exprs,
      function(e) activation_expr(activation, e, alpha = act_alpha),
      character(1L)
    )
  }
  state$all_exprs[[lname]] <- stats::setNames(conv_exprs, conv_nms)
  assign(lname, conv_nms, envir = expr_reg)
  invisible(NULL)
}
