# Generic standalone Activation layer handler.
#
# ReLU-family activation layer handlers (PReLU / LeakyReLU / ELU / ReLU)
# live in `model-keras-activation-relu-family.R`.
# Standalone Softmax layer + the shared softmax-block emitter live in
# `model-keras-activation-softmax.R`.
#
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

.k3_activation <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Standalone Activation layer: apply activation function to inbound expressions
  # NOTE: must be checked BEFORE the generic softmax branch is unnecessary —
  # the !grepl("softmax") guard prevents Keras3 Softmax() layers (whose class
  # path contains "activation") from being silently treated as linear identity.
  inbound <- topo_map[[lname]]
  if (is.null(inbound) || length(inbound) < 1L) {
    cli::cli_abort(
      "Activation layer {.val {lname}} has no inbound connections in model config."
    )
  }
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  act_cfg <- tryCatch(
    l$get_config()$activation,
    error = function(e) "linear"
  )
  activation <- if (is.null(act_cfg)) {
    "linear"
  } else if (is.list(act_cfg)) {
    tolower(as.character(act_cfg$class_name)[1L])
  } else {
    tolower(as.character(act_cfg)[1L])
  }
  if (!nzchar(activation)) {
    activation <- "linear"
  }
  if (activation %in% c("softmax", "log_softmax")) {
    .check_softmax_axis(l, lname)
    .k3_emit_softmax_block(
      in_exprs = in_exprs,
      lname = lname,
      expr_reg = expr_reg,
      state = state,
      nm_prefix = "orbital_act_sm_",
      unit_pfx = paste0("orbital_act_", lname, "_h"),
      log_form = activation == "log_softmax"
    )
  } else {
    unit_names <- paste0(
      "orbital_act_",
      lname,
      "_h",
      seq_along(in_exprs)
    )
    act_alpha <- if (is.list(act_cfg) && !is.null(act_cfg[["config"]])) {
      act_cfg[["config"]][["alpha"]]
    } else {
      NULL
    }
    act_default_value <- if (
      is.list(act_cfg) && !is.null(act_cfg[["config"]])
    ) {
      act_cfg[["config"]][["default_value"]] %||% 0
    } else {
      0
    }
    act_exprs <- vapply(
      in_exprs,
      function(e) {
        activation_expr(
          activation,
          e,
          alpha = act_alpha,
          default_value = act_default_value
        )
      },
      character(1)
    )
    state$all_exprs[[lname]] <- stats::setNames(act_exprs, unit_names)
    assign(lname, unit_names, envir = expr_reg)
  }
  invisible(NULL)
}
