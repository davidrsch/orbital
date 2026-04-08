# Handler function for the BatchNormalization layer.
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

.k3_batchnorm <- function(
    l,
    lname,
    topo_map,
    expr_reg,
    state,
    weight_map,
    output_layer_names,
    last_dense
) {
    # BatchNormalization: ((x - mean) / sqrt(var + eps)) * gamma + beta
    inbound <- topo_map[[lname]]
    in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
    wts <- l$get_weights() # gamma, beta, moving_mean, moving_var
    gamma <- as.numeric(wts[[1L]])
    beta <- as.numeric(wts[[2L]])
    mn <- as.numeric(wts[[3L]])
    vr <- as.numeric(wts[[4L]])
    eps <- tryCatch(as.numeric(l$epsilon), error = function(e) 1e-3)
    unit_names <- paste0(
        "orbital_bn_",
        lname,
        "_h",
        seq_along(in_exprs)
    )
    bn_exprs <- vapply(
        seq_along(in_exprs),
        function(i) {
            xe <- backtick(in_exprs[[i]])
            glue::glue(
                "(({xe} - {format_numeric(mn[i])}) /",
                " sqrt({format_numeric(vr[i])} + {format_numeric(eps)}))",
                " * {format_numeric(gamma[i])} + {format_numeric(beta[i])}"
            )
        },
        character(1)
    )
    state$all_exprs[[lname]] <- stats::setNames(bn_exprs, unit_names)
    assign(lname, unit_names, envir = expr_reg)
    invisible(NULL)
}
