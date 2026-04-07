# Handler functions for Dense / EinsumDense layers.
# Called by orbital_keras_dag_impl() in model-keras.R.

.k3_einsumdense <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # EinsumDense ─ generalised einsum projection; only Dense-equivalent equations.
  # Supported:
  #   "ab,bc->ac"   : input (B,A), kernel (A,C), output (B,C)  — simple Dense
  #   "...b,bc->...c": same with NumPy ellipsis                 — simple Dense
  #   "abc,cd->abd" : input (B,T,C), kernel (C,D), output (B,T,D) — time-dist Dense
  inbound <- topo_map[[lname]]
  if (is.null(inbound) || length(inbound) < 1L) {
    cli::cli_abort(
      "EinsumDense layer {.val {lname}} has no inbound connections in model config."
    )
  }
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  wts <- l$get_weights()
  if (length(wts) < 1L) {
    cli::cli_abort("EinsumDense layer {.val {lname}} has no weights.")
  }
  cfg_l <- tryCatch(l$get_config(), error = function(e) list())
  equation_raw <- tryCatch(
    as.character(cfg_l$equation),
    error = function(e) ""
  )
  equation <- gsub("\\s+", "", equation_raw)

  activation_config <- tryCatch(cfg_l$activation, error = function(e) NULL)
  activation <- if (is.null(activation_config)) {
    "linear"
  } else if (is.list(activation_config)) {
    tolower(as.character(activation_config$class_name)[1L])
  } else {
    tolower(as.character(activation_config)[1L])
  }
  if (!nzchar(activation)) {
    activation <- "linear"
  }
  act_alpha <- if (
    is.list(activation_config) &&
      !is.null(activation_config[["config"]])
  ) {
    activation_config[["config"]][["alpha"]]
  } else {
    NULL
  }

  kern <- wts[[1L]]
  bias_v_raw <- if (length(wts) >= 2L) as.numeric(wts[[2L]]) else NULL

  if (equation %in% c("ab,bc->ac", "...b,bc->...c")) {
    # Simple Dense: kernel (A, C); treat like Dense.
    n_out <- ncol(kern)
    bias_use <- if (is.null(bias_v_raw)) numeric(n_out) else bias_v_raw
    pre_act <- build_mlp_pre_act(t(kern), bias_use, in_exprs)
    act_exprs <- vapply(
      pre_act,
      function(z) activation_expr(activation, z, alpha = act_alpha),
      character(1L)
    )
    unit_names <- paste0(
      "orbital_einsumdense_",
      lname,
      "_h",
      seq_along(act_exprs)
    )
    state$all_exprs[[lname]] <- stats::setNames(act_exprs, unit_names)
    assign(lname, unit_names, envir = expr_reg)
  } else if (equation == "abc,cd->abd") {
    # Time-distributed Dense: kernel (C, D); apply per spatial position.
    C_last <- nrow(kern)
    D_out <- ncol(kern)
    T_len <- as.integer(length(in_exprs) / C_last)
    bias_use <- if (is.null(bias_v_raw)) numeric(D_out) else bias_v_raw
    k_T <- t(kern) # (D_out, C_last) for build_mlp_pre_act
    out_nms <- character(0L)
    out_exprs <- character(0L)
    for (p in seq_len(T_len)) {
      in_slice <- in_exprs[((p - 1L) * C_last + 1L):(p * C_last)]
      pre_act <- build_mlp_pre_act(k_T, bias_use, in_slice)
      ae <- vapply(
        pre_act,
        function(z) activation_expr(activation, z, alpha = act_alpha),
        character(1L)
      )
      nms <- paste0(
        "orbital_einsumdense_",
        lname,
        "_t",
        p,
        "_h",
        seq_len(D_out)
      )
      out_nms <- c(out_nms, nms)
      out_exprs <- c(out_exprs, ae)
    }
    state$all_exprs[[lname]] <- stats::setNames(out_exprs, out_nms)
    assign(lname, out_nms, envir = expr_reg)
  } else {
    cli::cli_abort(c(
      "EinsumDense layer {.val {lname}}: equation {.val {equation}} is not supported by orbital.",
      "i" = "Supported equations: 'ab,bc->ac', '...b,bc->...c', 'abc,cd->abd'."
    ))
  }
  invisible(NULL)
}


.k3_dense <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  inbound <- topo_map[[lname]]
  if (is.null(inbound) || length(inbound) < 1L) {
    cli::cli_abort(
      "Dense layer {.val {lname}} has no inbound connections in model config."
    )
  }
  in_names <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  wt <- weight_map[[lname]]
  pre_act <- build_mlp_pre_act(wt$kernel, wt$bias, in_names)

  activation_config <- tryCatch(
    l$get_config()$activation,
    error = function(e) NULL
  )
  activation <- if (is.null(activation_config)) {
    "linear"
  } else if (is.list(activation_config)) {
    tolower(as.character(activation_config$class_name)[1L])
  } else {
    tolower(as.character(activation_config)[1L])
  }
  if (!nzchar(activation)) {
    activation <- "linear"
  }
  act_alpha <- if (
    is.list(activation_config) &&
      !is.null(activation_config[["config"]])
  ) {
    activation_config[["config"]][["alpha"]]
  } else {
    NULL
  }

  if (lname %in% output_layer_names) {
    if (!activation %in% c("linear", "softmax", "log_softmax", "sigmoid")) {
      cli::cli_warn(c(
        "Dense output layer {.val {lname}} has activation = {.val {activation}} which orbital will override.",
        "i" = "orbital applies its own output transform based on {.arg mode}; the Keras activation on the output layer is ignored."
      ))
    }
    state$out_pre_act_map[[lname]] <- pre_act
    if (lname == last_dense) state$out_pre_act <- pre_act
  } else {
    act_exprs <- vapply(
      pre_act,
      function(z) {
        activation_expr(activation, z, alpha = act_alpha)
      },
      character(1)
    )
    unit_names <- paste0(
      "orbital_mlp_",
      lname,
      "_h",
      seq_along(act_exprs)
    )
    state$all_exprs[[lname]] <- stats::setNames(act_exprs, unit_names)
    assign(lname, unit_names, envir = expr_reg)
  }
  invisible(NULL)
}
