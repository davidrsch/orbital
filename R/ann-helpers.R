# Helper functions for artificial neural network (MLP) model implementations
# Used by model-nnet.R, model-brulee.R, model-keras.R, model-h2o.R, model-kerasnip.R

activation_expr <- function(activation, x_expr) {
    switch(
        activation,
        "linear" = as.character(x_expr),
        "relu" = ,
        "Rectifier" = ,
        "RectifierWithDropout" = glue::glue(
            "dplyr::if_else({x_expr} > 0, {x_expr}, 0)"
        ),
        "sigmoid" = glue::glue("1 / (1 + exp(-({x_expr})))"),
        "tanh" = ,
        "Tanh" = ,
        "TanhWithDropout" = glue::glue("tanh({x_expr})"),
        "elu" = ,
        "celu" = glue::glue(
            "dplyr::if_else({x_expr} >= 0, {x_expr}, exp({x_expr}) - 1)"
        ),
        "selu" = glue::glue(
            "dplyr::if_else({x_expr} > 0, 1.0507009873554805 * {x_expr}, 1.7580993408473766 * (exp({x_expr}) - 1))"
        ),
        "gelu" = glue::glue(
            "{x_expr} * 0.5 * (1 + tanh(({x_expr} + 0.044715 * {x_expr}^3) * 0.7978845608028654))"
        ),
        "hardshrink" = glue::glue(
            "dplyr::if_else(abs({x_expr}) > 0.5, {x_expr}, 0)"
        ),
        "hardsigmoid" = glue::glue(
            "dplyr::if_else({x_expr} <= -3, 0, dplyr::if_else({x_expr} >= 3, 1, {x_expr} / 6 + 0.5))"
        ),
        "hardtanh" = glue::glue(
            "dplyr::if_else({x_expr} < -1, -1, dplyr::if_else({x_expr} > 1, 1, {x_expr}))"
        ),
        "leaky_relu" = glue::glue(
            "dplyr::if_else({x_expr} >= 0, {x_expr}, 0.01 * {x_expr})"
        ),
        "log_sigmoid" = glue::glue("log(1 / (1 + exp(-({x_expr}))))"),
        "relu6" = glue::glue(
            "dplyr::if_else({x_expr} < 0, 0, dplyr::if_else({x_expr} > 6, 6, {x_expr}))"
        ),
        "rrelu" = glue::glue(
            "dplyr::if_else({x_expr} >= 0, {x_expr}, 0.22916666666666666 * {x_expr})"
        ),
        "silu" = ,
        "swish" = glue::glue("{x_expr} * (1 / (1 + exp(-({x_expr}))))"),
        "softplus" = glue::glue("log(1 + exp({x_expr}))"),
        "mish" = glue::glue("{x_expr} * tanh(log(1 + exp({x_expr})))"),
        "softshrink" = glue::glue(
            "dplyr::if_else({x_expr} > 0.5, {x_expr} - 0.5, dplyr::if_else({x_expr} < -0.5, {x_expr} + 0.5, 0))"
        ),
        "softsign" = glue::glue("{x_expr} / (1 + abs({x_expr}))"),
        "tanhshrink" = glue::glue("{x_expr} - tanh({x_expr})"),
        cli::cli_abort(
            "Activation function {.val {activation}} is not supported by orbital."
        )
    )
}

build_mlp_pre_act <- function(weight_mat, biases, input_names) {
    n_out <- nrow(weight_mat)
    vapply(
        seq_len(n_out),
        function(i) {
            terms <- format_numeric(biases[i])
            for (j in seq_along(input_names)) {
                w <- weight_mat[i, j]
                if (w == 0) {
                    next
                }
                terms <- c(
                    terms,
                    paste0(
                        "(",
                        backtick(input_names[j]),
                        " * ",
                        format_numeric(w),
                        ")"
                    )
                )
            }
            paste(terms, collapse = " + ")
        },
        character(1)
    )
}
