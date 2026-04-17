# Handler function for Conv1DTranspose (transposed 1-D convolution).
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

.k3_conv1dtranspose <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Conv1DTranspose ─ transposed (fractionally-strided) 1-D convolution.
  # Keras weight layout (same as Conv1D):
  #   kernel : (kernel_size, in_channels, filters)  i.e. (kW, C_in, C_out)
  #   bias   : (filters,)  [optional]
  # Dilations > 1 are not supported (raise cli_abort).
  # Padding: "valid" or "same".
  # W_out formula:
  #   valid: W_out = (W_in - 1) * stride + kW
  #   same:  W_out = W_in * stride
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

  wts <- .k3_get_weights(l, lname, required = 1L, names = c("kernel", "bias"))
  kern <- wts[[1L]] # (kW, C_in, C_out)
  k_w <- dim(kern)[1L]
  c_in <- dim(kern)[2L]
  c_out <- dim(kern)[3L]
  bias_v <- if (length(wts) >= 2L) as.numeric(wts[[2L]]) else numeric(c_out)

  cfg_l <- tryCatch(l$get_config(), error = function(e) list())
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

  if (dilation > 1L) {
    cli::cli_abort(
      "Keras Conv1DTranspose layer {.val {lname}}: dilation_rate > 1 is not supported."
    )
  }

  n_total <- length(in_exprs)
  w_in <- as.integer(n_total / c_in)

  if (pad_type == "same") {
    w_out <- w_in * stride
    pad_total <- max(0L, k_w - stride)
    pad_l <- as.integer(floor(pad_total / 2L))
  } else {
    # "valid"
    w_out <- (w_in - 1L) * stride + k_w
    pad_l <- 0L
  }

  tconv_nms <- character(0L)
  tconv_exprs <- character(0L)

  # For each output position q (0-indexed) and output channel f,
  # gather all input (p, k, c) triples that contribute:
  #   out[q, f] = sum_{p,k,c} in[p, c] * kern[k, c, f]
  # where p * stride + k - pad_l == q  (dilation=1)
  for (q in seq_len(w_out)) {
    q0 <- q - 1L
    for (f in seq_len(c_out)) {
      terms <- character(0L)
      for (k in seq_len(k_w)) {
        k0 <- k - 1L
        # p * stride = q0 + pad_l - k0
        num <- q0 + pad_l - k0
        if (num >= 0L && num %% stride == 0L) {
          p0 <- as.integer(num / stride)
          if (p0 >= 0L && p0 < w_in) {
            for (c in seq_len(c_in)) {
              feat_nm <- in_exprs[p0 * c_in + c]
              wt_val <- format_numeric(kern[k, c, f])
              terms <- c(
                terms,
                paste0(
                  "(",
                  backtick(feat_nm),
                  " * ",
                  wt_val,
                  ")"
                )
              )
            }
          }
        }
      }
      b_str <- format_numeric(bias_v[f])
      expr_str <- if (length(terms) == 0L) {
        b_str
      } else {
        paste0("(", paste(c(terms, b_str), collapse = " + "), ")")
      }
      nm <- paste0("orbital_tconv_", lname, "_q", q, "_f", f)
      tconv_nms <- c(tconv_nms, nm)
      tconv_exprs <- c(tconv_exprs, expr_str)
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
    tconv_exprs <- vapply(
      tconv_exprs,
      function(e) activation_expr(activation, e, alpha = act_alpha),
      character(1L)
    )
  }
  state$all_exprs[[lname]] <- stats::setNames(tconv_exprs, tconv_nms)
  assign(lname, tconv_nms, envir = expr_reg)
  invisible(NULL)
}
