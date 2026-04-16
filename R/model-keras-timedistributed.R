# TimeDistributed layer handler for the orbital Keras DAG backend.

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
        function(z) {
          activation_expr(activation_td, z, alpha = act_alpha_td)
        },
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
  } else if (grepl("conv1d", inner_cls, ignore.case = TRUE)) {
    inner_wts <- inner$get_weights()
    if (length(inner_wts) < 1L) {
      cli::cli_abort(
        "Keras TimeDistributed(Conv1D) layer {.val {lname}}: no weights found."
      )
    }
    # Keras3 Conv1D kernel shape: (kW, C_in, C_out)
    kern_raw <- inner_wts[[1L]]
    k_w <- dim(kern_raw)[1L]
    if (k_w != 1L) {
      cli::cli_abort(
        "Keras TimeDistributed(Conv1D) layer {.val {lname}}: kernel_size={k_w} > 1 is not supported."
      )
    }
    # Squeeze kW=1: (C_in, C_out) -> transpose to (C_out, C_in)
    kern_td_conv <- t(kern_raw[1L, , , drop = TRUE])
    c_out_conv <- nrow(kern_td_conv)
    c_in_conv <- ncol(kern_td_conv)
    bias_td_conv <- if (length(inner_wts) >= 2L) {
      as.numeric(inner_wts[[2L]])
    } else {
      numeric(c_out_conv)
    }
    act_conv_cfg <- tryCatch(
      inner$get_config()$activation,
      error = function(e) NULL
    )
    act_conv <- if (is.null(act_conv_cfg)) {
      "linear"
    } else if (is.list(act_conv_cfg)) {
      tolower(as.character(act_conv_cfg$class_name)[1L])
    } else {
      tolower(as.character(act_conv_cfg)[1L])
    }
    if (!nzchar(act_conv)) {
      act_conv <- "linear"
    }
    T_steps_conv <- as.integer(length(in_exprs) / c_in_conv)

    td_nms_conv <- character(0L)
    td_exprs_conv <- character(0L)
    for (t in seq_len(T_steps_conv)) {
      t_in_conv <- in_exprs[((t - 1L) * c_in_conv + 1L):(t * c_in_conv)]
      pre_act_conv <- build_mlp_pre_act(
        kern_td_conv,
        bias_td_conv,
        t_in_conv
      )
      act_str_conv <- vapply(
        pre_act_conv,
        function(z) activation_expr(act_conv, z),
        character(1L)
      )
      step_nms_conv <- paste0(
        "orbital_td_",
        lname,
        "_t",
        t,
        "_h",
        seq_len(c_out_conv)
      )
      td_nms_conv <- c(td_nms_conv, step_nms_conv)
      td_exprs_conv <- c(td_exprs_conv, act_str_conv)
    }
    state$all_exprs[[lname]] <- stats::setNames(td_exprs_conv, td_nms_conv)
    assign(lname, td_nms_conv, envir = expr_reg)
  } else {
    cli::cli_abort(c(
      paste0(
        "Keras TimeDistributed layer {.val {lname}} wraps unsupported inner ",
        "layer type {.cls {inner_cls}}."
      ),
      "i" = "orbital currently supports TimeDistributed(Dense) and TimeDistributed(Conv1D) with kernel_size=1 only."
    ))
  }
  invisible(NULL)
}
