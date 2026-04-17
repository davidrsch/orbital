# Handler function for the LayerNormalization layer.
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

.k3_layernorm <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # LayerNormalization: per-row symbolic mean + var, then normalize
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  wts <- .k3_get_weights(l, lname, required = 2L, names = c("gamma", "beta"))
  gamma <- as.numeric(wts[[1L]])
  beta <- as.numeric(wts[[2L]])
  n_feat <- length(in_exprs)
  eps <- tryCatch(as.numeric(l$epsilon), error = function(e) 1e-3)
  expr_bt <- backtick(in_exprs)
  mean_nm <- paste0("orbital_ln_mean_", lname)
  var_nm <- paste0("orbital_ln_var_", lname)
  mean_expr <- paste0(
    "(",
    paste(expr_bt, collapse = " + "),
    ") / ",
    n_feat
  )
  var_parts <- paste0("(", expr_bt, " - `", mean_nm, "`)^2")
  var_expr <- paste0(
    "(",
    paste(var_parts, collapse = " + "),
    ") / ",
    n_feat
  )
  unit_names <- paste0(
    "orbital_ln_",
    lname,
    "_h",
    seq_along(in_exprs)
  )
  norm_exprs <- vapply(
    seq_along(in_exprs),
    function(i) {
      xe <- backtick(in_exprs[[i]])
      glue::glue(
        "(({xe} - `{mean_nm}`) / sqrt(`{var_nm}` + {format_numeric(eps)}))",
        " * {format_numeric(gamma[i])} + {format_numeric(beta[i])}"
      )
    },
    character(1)
  )
  state$all_exprs[[lname]] <- c(
    stats::setNames(mean_expr, mean_nm),
    stats::setNames(var_expr, var_nm),
    stats::setNames(norm_exprs, unit_names)
  )
  assign(lname, unit_names, envir = expr_reg)
  invisible(NULL)
}
