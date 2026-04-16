# Handler for the Rescaling preprocessing layer.
# Called by orbital_keras_dag_impl() via .keras_dag_dispatch() in
# model-keras-dag-dispatch.R.
#
# Keras 3 Rescaling applies: y = x * scale + offset
# where scale and offset are stored in the layer config.

.k3_rescaling <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  cfg <- l$get_config()
  scale <- tryCatch(as.numeric(cfg$scale), error = function(e) 1.0)
  offset <- tryCatch(as.numeric(cfg$offset), error = function(e) 0.0)

  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

  out_names <- paste0("orbital_rescaling_", lname, "_h", seq_along(in_exprs))

  out_exprs <- vapply(
    seq_along(in_exprs),
    function(i) {
      xe <- backtick(in_exprs[[i]])
      as.character(glue::glue(
        "({xe}) * {format_numeric(scale)} + {format_numeric(offset)}"
      ))
    },
    character(1)
  )

  state$all_exprs[[lname]] <- stats::setNames(out_exprs, out_names)
  assign(lname, out_names, envir = expr_reg)
  invisible(NULL)
}
