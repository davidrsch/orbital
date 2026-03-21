# orbital methods for all keras-based models (legacy keras R package and keras3/kerasnip)

# Internal: parse inbound layer names from one layer's config entry.
# Handles keras3 new format ({args: [{keras_history: ["name", 0, 0]}]})
# and old keras format ([[["name", 0, 0], ...]]).
.keras_parse_inbound <- function(layer_cfg) {
    nodes <- layer_cfg[["inbound_nodes"]]
    if (is.null(nodes) || length(nodes) == 0L) {
        return(character(0L))
    }

    first_node <- nodes[[1L]]

    # keras3 format: first_node = list(args = list(...), kwargs = list())
    args <- first_node[["args"]]
    if (!is.null(args) && is.list(args)) {
        res <- vapply(
            args,
            function(arg) {
                hist <- arg[["keras_history"]]
                if (is.null(hist)) {
                    return(NA_character_)
                }
                lyr <- hist[[1L]]
                if (is.character(lyr)) {
                    lyr
                } else {
                    tryCatch(lyr$name, error = function(e) {
                        as.character(lyr)[1L]
                    })
                }
            },
            character(1L)
        )
        return(res[!is.na(res)])
    }

    # Old keras format: first_node is a list of [layer_name, node_idx, tensor_idx] triples
    if (
        is.list(first_node) &&
            length(first_node) > 0L &&
            is.list(first_node[[1L]])
    ) {
        return(vapply(
            first_node,
            function(spec) {
                n <- spec[[1L]]
                if (is.character(n)) n else as.character(n)[1L]
            },
            character(1L)
        ))
    }

    character(0L)
}


# Internal: topological/DAG traversal for Functional API models with merge layers.
orbital_keras_dag_impl <- function(
    x,
    mode,
    type,
    lvl,
    prefix,
    feature_names,
    all_weights,
    all_layers
) {
    # Build Dense weight map: layer_name → list(kernel, bias)
    # Use per-layer get_weights() so BatchNorm/PReLU weights don't skew indices.
    weight_map <- list()
    for (l in all_layers) {
        if (grepl("dense", tolower(class(l)[1L]))) {
            wts <- l$get_weights()
            if (length(wts) >= 2L) {
                weight_map[[l$name]] <- list(
                    kernel = t(wts[[1L]]),
                    bias = as.numeric(wts[[2L]])
                )
            }
        }
    }
    last_dense <- if (length(weight_map) > 0L) {
        tail(names(weight_map), 1L)
    } else {
        ""
    }

    n_in <- ncol(weight_map[[names(weight_map)[1L]]]$kernel)
    input_names <- if (!is.null(feature_names)) {
        feature_names
    } else {
        paste0("orbital_feature_", seq_len(n_in))
    }

    # Parse layer topology from model config
    cfg <- tryCatch(x$get_config(), error = function(e) NULL)
    layers_cfg <- if (!is.null(cfg)) {
        if (!is.null(cfg$config) && !is.null(cfg$config$layers)) {
            cfg$config$layers
        } else if (!is.null(cfg$layers)) {
            cfg$layers
        } else {
            list()
        }
    } else {
        list()
    }

    topo_map <- vector("list", length(layers_cfg))
    for (k in seq_along(layers_cfg)) {
        lc <- layers_cfg[[k]]
        lname <- tryCatch(
            {
                n <- if (!is.null(lc$config) && !is.null(lc$config$name)) {
                    lc$config$name
                } else {
                    lc$name
                }
                as.character(n)[1L]
            },
            error = function(e) ""
        )
        if (nzchar(lname)) {
            topo_map[[lname]] <- .keras_parse_inbound(lc)
        }
    }

    # Expression register: layer_name → character vector of expression names
    expr_reg <- new.env(hash = TRUE, parent = emptyenv())
    all_exprs <- list()
    out_pre_act <- NULL

    for (l in all_layers) {
        cls_orig <- class(l)[1L]
        cls <- tolower(cls_orig)
        lname <- l$name

        if (grepl("input", cls)) {
            assign(lname, input_names, envir = expr_reg)
        } else if (grepl("dense", cls)) {
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

            if (lname == last_dense) {
                out_pre_act <- pre_act
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
                all_exprs[[lname]] <- stats::setNames(act_exprs, unit_names)
                assign(lname, unit_names, envir = expr_reg)
            }
        } else if (grepl("\\badd\\b", cls, perl = TRUE)) {
            # Element-wise Add: supports skip / residual connections
            inbound <- topo_map[[lname]]
            if (is.null(inbound) || length(inbound) != 2L) {
                cli::cli_abort(
                    "Keras Add layer {.val {lname}} must have exactly 2 inbound inputs, got {length(inbound)}."
                )
            }
            exprs_a <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
            exprs_b <- get(inbound[2L], envir = expr_reg, inherits = FALSE)
            if (length(exprs_a) != length(exprs_b)) {
                cli::cli_abort(
                    "Keras Add layer {.val {lname}}: both inputs must have the same width ({length(exprs_a)} vs {length(exprs_b)})."
                )
            }
            add_names <- paste0("orbital_", lname, "_h", seq_along(exprs_a))
            add_exprs <- paste0(
                "(",
                backtick(exprs_a),
                " + ",
                backtick(exprs_b),
                ")"
            )
            all_exprs[[lname]] <- stats::setNames(add_exprs, add_names)
            assign(lname, add_names, envir = expr_reg)
        } else if (grepl("concatenate", cls)) {
            # Concatenate: merge expression-name vectors from all inbound branches
            inbound <- topo_map[[lname]]
            if (is.null(inbound) || length(inbound) < 2L) {
                cli::cli_abort(
                    "Keras Concatenate layer {.val {lname}} must have at least 2 inbound inputs."
                )
            }
            combined_names <- unlist(lapply(inbound, function(nm) {
                get(nm, envir = expr_reg, inherits = FALSE)
            }))
            assign(lname, combined_names, envir = expr_reg)
        } else if (grepl("batchnorm", cls)) {
            # BatchNormalization: ((x - mean) / sqrt(var + eps)) * gamma + beta
            inbound <- topo_map[[lname]]
            in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
            wts <- l$get_weights() # gamma, beta, moving_mean, moving_var
            gamma <- as.numeric(wts[[1L]])
            beta <- as.numeric(wts[[2L]])
            mn <- as.numeric(wts[[3L]])
            vr <- as.numeric(wts[[4L]])
            eps <- tryCatch(as.numeric(l$epsilon), error = function(e) 1e-5)
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
                        "(({xe} - {format_numeric(mn[i])}) / sqrt({format_numeric(vr[i])} + {format_numeric(eps)})) * {format_numeric(gamma[i])} + {format_numeric(beta[i])}"
                    )
                },
                character(1)
            )
            all_exprs[[lname]] <- stats::setNames(bn_exprs, unit_names)
            assign(lname, unit_names, envir = expr_reg)
        } else if (grepl("layernorm", cls)) {
            # LayerNormalization: per-row symbolic mean + var, then normalize
            inbound <- topo_map[[lname]]
            in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
            wts <- l$get_weights() # gamma, beta
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
                        "(({xe} - `{mean_nm}`) / sqrt(`{var_nm}` + {format_numeric(eps)})) * {format_numeric(gamma[i])} + {format_numeric(beta[i])}"
                    )
                },
                character(1)
            )
            all_exprs[[lname]] <- c(
                stats::setNames(mean_expr, mean_nm),
                stats::setNames(var_expr, var_nm),
                stats::setNames(norm_exprs, unit_names)
            )
            assign(lname, unit_names, envir = expr_reg)
        } else if (grepl("prelu", cls)) {
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
            all_exprs[[lname]] <- stats::setNames(prelu_exprs, unit_names)
            assign(lname, unit_names, envir = expr_reg)
        } else if (grepl("dropout|flatten|reshape|activation", cls)) {
            # Inference-transparent pass-through layers
            inbound <- topo_map[[lname]]
            if (
                !is.null(inbound) &&
                    length(inbound) >= 1L &&
                    exists(inbound[1L], envir = expr_reg, inherits = FALSE)
            ) {
                assign(
                    lname,
                    get(inbound[1L], envir = expr_reg, inherits = FALSE),
                    envir = expr_reg
                )
            }
        } else {
            cli::cli_abort(c(
                "Unsupported layer type in Keras Functional model: {.cls {cls_orig}}.",
                "i" = "orbital supports: Dense, Add, Concatenate, BatchNormalization, LayerNormalization, PReLU, Dropout, Flatten, Reshape, Activation.",
                "i" = "Please file an issue: {.url https://github.com/davidrsch/orbital/issues/14}"
            ))
        }
    }

    if (is.null(out_pre_act)) {
        cli::cli_abort(
            "Could not identify the output Dense layer in the Keras model."
        )
    }

    hidden_exprs <- unlist(all_exprs, use.names = TRUE)
    n_out <- length(out_pre_act)

    if (mode == "regression") {
        c(hidden_exprs, stats::setNames(out_pre_act[1L], prefix))
    } else if (n_out == 1L) {
        c(
            hidden_exprs,
            binary_from_prob(
                activation_expr("sigmoid", out_pre_act[1L]),
                type,
                lvl
            )
        )
    } else {
        c(
            hidden_exprs,
            multiclass_from_logits(stats::setNames(out_pre_act, lvl), type, lvl)
        )
    }
}


orbital_keras_impl <- function(
    x,
    mode,
    type,
    lvl,
    prefix,
    feature_names = NULL
) {
    all_weights <- x$get_weights()
    all_layers <- x$layers

    # Detect layers that require DAG traversal (merge, normalisation, learned activations).
    non_linear_layers <- Filter(
        function(l) {
            cls <- tolower(class(l)[1L])
            grepl(
                "\\badd\\b|concatenate|batchnorm|layernorm|prelu",
                cls,
                perl = TRUE
            ) &&
                !grepl(
                    "dense|input|flatten|reshape|activation|dropout",
                    cls
                )
        },
        all_layers
    )

    if (length(non_linear_layers) > 0L) {
        return(
            orbital_keras_dag_impl(
                x,
                mode,
                type,
                lvl,
                prefix,
                feature_names,
                all_weights,
                all_layers
            )
        )
    }

    # Linear model: optimised sequential traversal (no topology introspection needed)
    n_dense <- length(all_weights) / 2L # each Dense layer has kernel + bias

    dense_layers <- Filter(
        function(l) grepl("dense", tolower(class(l)[1L])),
        all_layers
    )

    n_in <- ncol(all_weights[[1L]])
    input_names <- if (!is.null(feature_names)) {
        feature_names
    } else {
        paste0("orbital_feature_", seq_len(n_in))
    }

    all_exprs <- list()
    current_names <- input_names

    for (i in seq_len(n_dense)) {
        kernel <- t(all_weights[[2L * i - 1L]]) # (n_out_i x n_in_i)
        bias <- as.numeric(all_weights[[2L * i]])

        activation_config <- tryCatch(
            dense_layers[[i]]$get_config()$activation,
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

        pre_act <- build_mlp_pre_act(kernel, bias, current_names)

        if (i < n_dense) {
            act_exprs <- vapply(
                pre_act,
                function(z) activation_expr(activation, z, alpha = act_alpha),
                character(1)
            )
            layer_names <- paste0(
                "orbital_mlp_l",
                i,
                "_h",
                seq_len(nrow(kernel))
            )
            all_exprs[[i]] <- stats::setNames(act_exprs, layer_names)
            current_names <- layer_names
        } else {
            out_pre_act <- pre_act
        }
    }

    hidden_exprs <- unlist(all_exprs, use.names = TRUE)
    n_out <- length(out_pre_act)

    if (mode == "regression") {
        c(hidden_exprs, stats::setNames(out_pre_act[1L], prefix))
    } else if (n_out == 1L) {
        sigmoid_expr <- activation_expr("sigmoid", out_pre_act[1L])
        c(hidden_exprs, binary_from_prob(sigmoid_expr, type, lvl))
    } else {
        logit_exprs <- stats::setNames(out_pre_act, lvl)
        c(hidden_exprs, multiclass_from_logits(logit_exprs, type, lvl))
    }
}


#' @method orbital keras.engine.sequential.Sequential
#' @export
orbital.keras.engine.sequential.Sequential <- function(
    x,
    ...,
    mode = c("regression", "classification"),
    type = NULL,
    lvl = NULL,
    prefix = ".pred"
) {
    mode <- rlang::arg_match(mode)
    type <- default_type(type)
    extra <- list(...)
    orbital_keras_impl(
        x,
        mode = mode,
        type = type,
        lvl = lvl,
        prefix = prefix,
        feature_names = extra$feature_names
    )
}

#' @method orbital keras.src.models.sequential.Sequential
#' @export
orbital.keras.src.models.sequential.Sequential <- function(
    x,
    ...,
    mode = c("regression", "classification"),
    type = NULL,
    lvl = NULL,
    prefix = ".pred"
) {
    mode <- rlang::arg_match(mode)
    type <- default_type(type)
    extra <- list(...)
    orbital_keras_impl(
        x,
        mode = mode,
        type = type,
        lvl = lvl,
        prefix = prefix,
        feature_names = extra$feature_names
    )
}

#' @method orbital keras.src.models.functional.Functional
#' @export
orbital.keras.src.models.functional.Functional <- function(
    x,
    ...,
    mode = c("regression", "classification"),
    type = NULL,
    lvl = NULL,
    prefix = ".pred"
) {
    mode <- rlang::arg_match(mode)
    type <- default_type(type)
    extra <- list(...)
    orbital_keras_impl(
        x,
        mode = mode,
        type = type,
        lvl = lvl,
        prefix = prefix,
        feature_names = extra$feature_names
    )
}
