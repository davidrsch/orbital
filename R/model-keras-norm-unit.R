# Handler for the UnitNormalization layer.
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

.k3_unitnorm <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # UnitNormalization: L2-normalise each row across all features.
  # y_i = x_i / sqrt(max(x_1^2 + ... + x_n^2, 1e-7))
  # Matches Keras 3: keras.backend.epsilon() = 1e-7 (float32 default).
  # Only axis = -1 (normalise over feature dimension) is supported.
  unit_axis_raw <- tryCatch(
    as.integer(unlist(l$get_config()$axis)[[1L]]),
    error = function(e) -1L
  )
  if (is.na(unit_axis_raw)) {
    unit_axis_raw <- -1L
  }
  if (!unit_axis_raw %in% c(-1L, 1L)) {
    cli::cli_abort(
      c(
        "UnitNormalization layer {.val {lname}}: axis = {.val {unit_axis_raw}} is not supported.",
        "i" = "Only axis = -1 (feature-wise L2 normalisation) is supported by orbital."
      )
    )
  }
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  n_feat <- length(in_exprs)
  expr_bt <- backtick(in_exprs)
  norm_nm <- paste0("orbital_unitnorm_", lname, "_norm")
  sq_sum <- paste(paste0(expr_bt, "^2"), collapse = " + ")
  norm_expr <- paste0("sqrt(do.call(pmax, list(", sq_sum, ", 1e-7)))")
  unit_names <- .orb_col_nms("unitnorm", lname, "h", n_feat)
  unit_exprs <- vapply(
    seq_len(n_feat),
    function(i) paste0("(", expr_bt[i], " / `", norm_nm, "`)"),
    character(1L)
  )
  state$all_exprs[[lname]] <- c(
    stats::setNames(norm_expr, norm_nm),
    stats::setNames(unit_exprs, unit_names)
  )
  assign(lname, unit_names, envir = expr_reg)
  invisible(NULL)
}
