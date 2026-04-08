# Handler functions for global 1-D pooling layers.
# (GlobalAveragePooling1D, GlobalMaxPooling1D, GlobalSumPooling1D).
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

.k3_globalaveragepool <- function(
    l,
    lname,
    topo_map,
    expr_reg,
    state,
    weight_map,
    output_layer_names,
    last_dense
) {
    # GlobalAveragePooling1D: reduce feature columns to their row-wise mean
    inbound <- topo_map[[lname]]
    in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
    n_f <- length(in_exprs)
    expr_bt <- backtick(in_exprs)
    gap_nm <- paste0("orbital_gap_", lname, "_1")
    gap_expr <- paste0(
        "(",
        paste(expr_bt, collapse = " + "),
        ") / ",
        n_f
    )
    state$all_exprs[[lname]] <- stats::setNames(gap_expr, gap_nm)
    assign(lname, gap_nm, envir = expr_reg)
    invisible(NULL)
}


.k3_globalmaxpool <- function(
    l,
    lname,
    topo_map,
    expr_reg,
    state,
    weight_map,
    output_layer_names,
    last_dense
) {
    # GlobalMaxPooling1D: reduce feature columns to their row-wise max
    inbound <- topo_map[[lname]]
    in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
    expr_bt <- backtick(in_exprs)
    gmp_nm <- paste0("orbital_gmp_", lname, "_1")
    gmp_expr <- paste0(
        "do.call(pmax, list(",
        paste(expr_bt, collapse = ", "),
        "))"
    )
    state$all_exprs[[lname]] <- stats::setNames(gmp_expr, gmp_nm)
    assign(lname, gmp_nm, envir = expr_reg)
    invisible(NULL)
}


.k3_globalsumpooling <- function(
    l,
    lname,
    topo_map,
    expr_reg,
    state,
    weight_map,
    output_layer_names,
    last_dense
) {
    # GlobalSumPooling1D: row-wise sum of all feature columns.
    # NOTE: GlobalSumPooling1D is not a standard Keras 3 layer; it exists
    # only in keras_cv (keras-cv package). This branch is effectively dead
    # code for stock Keras 3 models. If you are using keras_cv, verify that
    # the class name it produces contains "globalsumpooling"; otherwise this
    # branch will never be reached.
    inbound <- topo_map[[lname]]
    in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
    expr_bt <- backtick(in_exprs)
    gsp_nm <- paste0("orbital_gsp_", lname, "_1")
    gsp_expr <- paste0(
        "(",
        paste(expr_bt, collapse = " + "),
        ")"
    )
    state$all_exprs[[lname]] <- stats::setNames(gsp_expr, gsp_nm)
    assign(lname, gsp_nm, envir = expr_reg)
    invisible(NULL)
}
