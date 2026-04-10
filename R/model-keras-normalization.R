# Handler function for the Normalization preprocessing layer.
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

.k3_normalization <- function(
    l,
    lname,
    topo_map,
    expr_reg,
    state,
    weight_map,
    output_layer_names,
    last_dense
) {
    # Normalization: (x - mean) / sqrt(variance + epsilon)
    # Weights from adapt(): wts[[1]] = adapt_mean, wts[[2]] = adapt_variance
    inbound <- topo_map[[lname]]
    in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
    wts <- l$get_weights()
    mn <- as.numeric(wts[[1L]])
    vr <- as.numeric(wts[[2L]])
    eps <- tryCatch(as.numeric(l$epsilon), error = function(e) 1e-3)
    unit_names <- paste0(
        "orbital_norm_",
        lname,
        "_h",
        seq_along(in_exprs)
    )
    norm_exprs <- vapply(
        seq_along(in_exprs),
        function(i) {
            xe <- backtick(in_exprs[[i]])
            glue::glue(
                "({xe} - {format_numeric(mn[i])}) /",
                " sqrt({format_numeric(vr[i])} + {format_numeric(eps)})"
            )
        },
        character(1L)
    )
    state$all_exprs[[lname]] <- stats::setNames(norm_exprs, unit_names)
    assign(lname, unit_names, envir = expr_reg)
    invisible(NULL)
}
