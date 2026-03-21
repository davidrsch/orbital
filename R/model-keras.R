# orbital methods for all keras-based models (legacy keras R package and keras3/kerasnip)

orbital_keras_impl <- function(
    x,
    mode,
    type,
    lvl,
    prefix,
    feature_names = NULL
) {
    all_weights <- x$get_weights()
    n_dense <- length(all_weights) / 2L # each Dense layer has kernel + bias

    # Get all Dense layers for activation introspection
    dense_layers <- Filter(
        function(l) grepl("dense", tolower(class(l)[1])),
        x$layers
    )

    # Input names
    n_in <- ncol(all_weights[[1L]])
    if (is.null(feature_names)) {
        input_names <- paste0("orbital_feature_", seq_len(n_in))
    } else {
        input_names <- feature_names
    }

    all_exprs <- list()
    current_names <- input_names

    for (i in seq_len(n_dense)) {
        kernel <- t(all_weights[[2L * i - 1L]]) # (n_out_i x n_in_i)
        bias <- as.numeric(all_weights[[2L * i]])

        # Get activation for this layer from config
        activation <- tryCatch(
            {
                cfg <- dense_layers[[i]]$get_config()
                act <- cfg$activation
                if (is.list(act)) {
                    act <- act$class_name
                }
                tolower(as.character(act))
            },
            error = function(e) "linear"
        )
        if (is.null(activation) || activation == "") {
            activation <- "linear"
        }

        pre_act <- build_mlp_pre_act(kernel, bias, current_names)

        if (i < n_dense) {
            # Hidden layer: apply activation
            act_exprs <- vapply(
                pre_act,
                function(z) activation_expr(activation, z),
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
            # Output layer: hold pre-activations; activation determined by mode
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
