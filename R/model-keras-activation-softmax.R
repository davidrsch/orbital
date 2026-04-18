# Standalone Softmax / log_softmax layer handlers, plus the shared
# emit-softmax-block helper used by both .k3_softmax() and the softmax /
# log_softmax branches of .k3_activation() in model-keras-activation-layers.R.

# Emit a max-stabilised softmax (or log_softmax) block.
#
# Subtracting the row-wise max before exp() prevents overflow for large logits
# and does not change the result (the shift cancels in numerator and
# denominator for softmax, and the additive shift cancels in log_softmax via
# the `log(sum(exp(x - max)))` form).
#
# Returns invisibly; mutates `state$all_exprs[[lname]]` and registers the
# unit-name vector in `expr_reg` under `lname`.
#
# Parameters
# ----------
# in_exprs  : character vector of inbound expression-column names
# lname     : the keras layer name (used for unique intermediate column names)
# nm_prefix : prefix used for intermediate column names
#             (e.g. "orbital_sm_" for standalone Softmax; "orbital_act_sm_"
#              for softmax inside an Activation layer)
# unit_pfx  : prefix for the per-unit output columns
#             (e.g. "orbital_sm_<lname>_h" or "orbital_act_<lname>_h")
# log_form  : if TRUE, emit log_softmax form; otherwise emit softmax.
.k3_emit_softmax_block <- function(
  in_exprs,
  lname,
  expr_reg,
  state,
  nm_prefix,
  unit_pfx,
  log_form = FALSE
) {
  expr_bt <- backtick(in_exprs)
  max_nm <- paste0(nm_prefix, "max_", lname)
  max_expr <- paste0(
    "do.call(pmax, list(",
    paste(expr_bt, collapse = ", "),
    "))"
  )
  sum_nm <- paste0(nm_prefix, "sum_", lname)
  sum_expr <- paste0(
    "(",
    paste0(
      "exp(",
      expr_bt,
      " - `",
      max_nm,
      "`)",
      collapse = " + "
    ),
    ")"
  )
  unit_names <- paste0(unit_pfx, seq_along(in_exprs))
  unit_exprs <- if (log_form) {
    vapply(
      seq_along(in_exprs),
      function(i) {
        paste0(
          "(",
          expr_bt[i],
          " - `",
          max_nm,
          "`) - log(`",
          sum_nm,
          "`)"
        )
      },
      character(1)
    )
  } else {
    vapply(
      seq_along(in_exprs),
      function(i) {
        paste0(
          "exp(",
          expr_bt[i],
          " - `",
          max_nm,
          "`) / `",
          sum_nm,
          "`"
        )
      },
      character(1)
    )
  }
  state$all_exprs[[lname]] <- c(
    stats::setNames(max_expr, max_nm),
    stats::setNames(sum_expr, sum_nm),
    stats::setNames(unit_exprs, unit_names)
  )
  assign(lname, unit_names, envir = expr_reg)
  invisible(NULL)
}

.k3_softmax <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Standalone Softmax layer: row-wise softmax normalisation (max-stabilised).
  # Only axis = -1 (the last/feature axis) is supported; any other axis would
  # normalise across a non-feature dimension that orbital's flat-column layout
  # cannot represent without silently producing wrong results.
  .check_softmax_axis(l, lname)
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  .k3_emit_softmax_block(
    in_exprs = in_exprs,
    lname = lname,
    expr_reg = expr_reg,
    state = state,
    nm_prefix = "orbital_sm_",
    unit_pfx = paste0("orbital_sm_", lname, "_h"),
    log_form = FALSE
  )
}
