# Handler functions for standalone activation layers (PReLU, LeakyReLU, ELU,
# ReLU, Activation). These were originally in model-keras-norm-standard.R
# alongside BatchNorm and LayerNorm.
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

.k3_prelu <- function(
    l,
    lname,
    topo_map,
    expr_reg,
    state,
    weight_map,
    output_layer_names,
    last_dense
) {
    # PReLU: per-channel learnable negative slope
    inbound <- topo_map[[lname]]
    in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
    alpha_w <- as.numeric(unlist(l$get_weights()))
    unit_names <- paste0(
        "orbital_prelu_",
        lname,
        "_h",
        seq_along(in_exprs)
    )
    prelu_exprs <- vapply(
        seq_along(in_exprs),
        function(i) {
            xe <- backtick(in_exprs[[i]])
            a <- format_numeric(alpha_w[min(i, length(alpha_w))])
            glue::glue("dplyr::if_else({xe} >= 0, {xe}, {a} * {xe})")
        },
        character(1)
    )
    state$all_exprs[[lname]] <- stats::setNames(prelu_exprs, unit_names)
    assign(lname, unit_names, envir = expr_reg)
    invisible(NULL)
}


.k3_leakyrelu <- function(
    l,
    lname,
    topo_map,
    expr_reg,
    state,
    weight_map,
    output_layer_names,
    last_dense
) {
    # Standalone LeakyReLU layer (keras.src.layers.activation.leaky_relu.*)
    # Must be checked BEFORE the generic \bactivation\b branch because the
    # module path contains the word "activation".
    inbound <- topo_map[[lname]]
    in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
    slope <- tryCatch(
        as.numeric(l$get_config()$negative_slope),
        error = function(e) 0.01
    )
    if (is.null(slope) || is.na(slope)) {
        slope <- 0.01
    }
    unit_names <- paste0(
        "orbital_leakyrelu_",
        lname,
        "_h",
        seq_along(in_exprs)
    )
    lr_exprs <- vapply(
        in_exprs,
        function(e) activation_expr("leaky_relu", e, alpha = slope),
        character(1)
    )
    state$all_exprs[[lname]] <- stats::setNames(lr_exprs, unit_names)
    assign(lname, unit_names, envir = expr_reg)
    invisible(NULL)
}


.k3_elu <- function(
    l,
    lname,
    topo_map,
    expr_reg,
    state,
    weight_map,
    output_layer_names,
    last_dense
) {
    # Standalone ELU layer (keras.src.layers.activation.elu.ELU).
    # The \belu\b word-boundary pattern does NOT match selu, celu, relu,
    # prelu, or leakyrelu, so this branch is safe to place after leakyrelu.
    inbound <- topo_map[[lname]]
    in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
    alpha_val <- tryCatch(
        as.numeric(l$get_config()$alpha),
        error = function(e) 1.0
    )
    if (is.null(alpha_val) || is.na(alpha_val)) {
        alpha_val <- 1.0
    }
    unit_names <- paste0(
        "orbital_elu_",
        lname,
        "_h",
        seq_along(in_exprs)
    )
    elu_exprs <- vapply(
        in_exprs,
        function(e) activation_expr("elu", e, alpha = alpha_val),
        character(1)
    )
    state$all_exprs[[lname]] <- stats::setNames(elu_exprs, unit_names)
    assign(lname, unit_names, envir = expr_reg)
    invisible(NULL)
}


.k3_relu <- function(
    l,
    lname,
    topo_map,
    expr_reg,
    state,
    weight_map,
    output_layer_names,
    last_dense
) {
    # Standalone ReLU layer (keras.layers.ReLU()): must be checked BEFORE the
    # generic \bactivation\b branch because the module path contains "activation".
    # Keras ReLU config: negative_slope (default 0), max_value (default NULL),
    # threshold (default 0).  Full formula:
    #   base = if_else(x >= threshold, x - threshold, neg_slope * (x - threshold))
    #   out  = if max_value set: if_else(base > max_value, max_value, base)
    inbound <- topo_map[[lname]]
    in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
    ns <- tryCatch(
        as.numeric(l$get_config()$negative_slope),
        error = function(e) 0.0
    )
    thr <- tryCatch(
        as.numeric(l$get_config()$threshold),
        error = function(e) 0.0
    )
    if (is.null(ns) || length(ns) == 0L || is.na(ns)) {
        ns <- 0.0
    }
    if (is.null(thr) || length(thr) == 0L || is.na(thr)) {
        thr <- 0.0
    }
    mv_raw <- tryCatch(l$get_config()$max_value, error = function(e) NULL)
    mv_finite <- !is.null(mv_raw) &&
        length(mv_raw) > 0L &&
        !is.na(suppressWarnings(as.numeric(mv_raw)[[1L]]))
    mv <- if (mv_finite) as.numeric(mv_raw)[[1L]] else NA_real_
    unit_names <- .orb_col_nms("relu", lname, "h", length(in_exprs))
    relu_exprs <- vapply(
        in_exprs,
        function(e) {
            xe <- backtick(e)
            thr_f <- format_numeric(thr)
            ns_f <- format_numeric(ns)
            base <- if (thr == 0.0 && ns == 0.0) {
                glue::glue("dplyr::if_else({xe} >= 0, {xe}, 0)")
            } else if (thr == 0.0) {
                glue::glue("dplyr::if_else({xe} >= 0, {xe}, {ns_f} * {xe})")
            } else {
                glue::glue(
                    "dplyr::if_else({xe} >= {thr_f}, {xe} - {thr_f}, {ns_f} * ({xe} - {thr_f}))"
                )
            }
            if (mv_finite) {
                mv_f <- format_numeric(mv)
                base <- glue::glue(
                    "dplyr::if_else({base} > {mv_f}, {mv_f}, {base})"
                )
            }
            base
        },
        character(1L)
    )
    state$all_exprs[[lname]] <- stats::setNames(relu_exprs, unit_names)
    assign(lname, unit_names, envir = expr_reg)
    invisible(NULL)
}


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
        expr_bt <- backtick(in_exprs)
        # Max-stabilisation: subtract max(logits) before exp() so that
        # exp() never overflows and results are numerically identical to
        # the unstabilised form (the shift cancels in both softmax and
        # log_softmax).
        max_nm <- paste0("orbital_act_sm_max_", lname)
        max_expr <- paste0(
            "do.call(pmax, list(",
            paste(expr_bt, collapse = ", "),
            "))"
        )
        sm_sum_nm <- paste0("orbital_act_sm_sum_", lname)
        sm_sum_expr <- paste0(
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
        unit_names <- paste0(
            "orbital_act_",
            lname,
            "_h",
            seq_along(in_exprs)
        )
        sm_exprs <- if (activation == "softmax") {
            vapply(
                seq_along(in_exprs),
                function(i) {
                    paste0(
                        "exp(",
                        expr_bt[i],
                        " - `",
                        max_nm,
                        "`) / `",
                        sm_sum_nm,
                        "`"
                    )
                },
                character(1)
            )
        } else {
            # log_softmax = (x_i - max) - log(sum(exp(x_j - max)))
            vapply(
                seq_along(in_exprs),
                function(i) {
                    paste0(
                        "(",
                        expr_bt[i],
                        " - `",
                        max_nm,
                        "`) - log(`",
                        sm_sum_nm,
                        "`)"
                    )
                },
                character(1)
            )
        }
        state$all_exprs[[lname]] <- c(
            stats::setNames(max_expr, max_nm),
            stats::setNames(sm_sum_expr, sm_sum_nm),
            stats::setNames(sm_exprs, unit_names)
        )
        assign(lname, unit_names, envir = expr_reg)
    } else {
        unit_names <- paste0(
            "orbital_act_",
            lname,
            "_h",
            seq_along(in_exprs)
        )
        act_exprs <- vapply(
            in_exprs,
            function(e) activation_expr(activation, e),
            character(1)
        )
        state$all_exprs[[lname]] <- stats::setNames(act_exprs, unit_names)
        assign(lname, unit_names, envir = expr_reg)
    }
    invisible(NULL)
}
