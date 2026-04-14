# qrnn backend for orbital
#
# qrnn::qrnn.fit() returns a plain list with no S3 class.  To enable
# UseMethod("orbital") dispatch, wrap the list with as_qrnn_fit():
#
#   m <- qrnn::qrnn.fit(...)
#   orbital(as_qrnn_fit(m))

#' Tag a raw qrnn model for use with orbital
#'
#' `qrnn::qrnn.fit()` returns a plain `list` with no S3 class.  There is no
#' standard parsnip engine for qrnn, so `orbital()` cannot automatically
#' dispatch on the raw list.  Call `as_qrnn_fit(m)` once to attach the
#' `"qrnn_fit"` sub-class, then pass the result to `orbital()`.
#'
#' @param x A list returned by [qrnn::qrnn.fit()].
#'
#' @returns `x` with `class` set to `c("qrnn_fit", "list")`.
#'
#' @seealso [orbital.qrnn_fit()]
#'
#' @export
as_qrnn_fit <- function(x) {
    if (!is.list(x)) {
        cli::cli_abort(
            "{.fn as_qrnn_fit} requires a list returned by {.fn qrnn::qrnn.fit}."
        )
    }
    required <- c("weights", "Th", "x.center", "x.scale", "y.center", "y.scale")
    missing <- setdiff(required, names(x))
    if (length(missing) > 0L) {
        cli::cli_abort(
            c(
                "Object passed to {.fn as_qrnn_fit} does not look like a qrnn fit.",
                "i" = "Missing fields: {.val {missing}}."
            )
        )
    }
    class(x) <- c("qrnn_fit", "list")
    x
}

#' Turn a qrnn model into an orbital object
#'
#' Generates dplyr/SQL-portable column expressions that replicate the forward
#' pass of a [qrnn::qrnn.fit()] model: input standardisation, hidden layer
#' with the trained activation, and output de-standardisation.
#'
#' Only single-quantile, single-hidden-layer, non-additive qrnn models
#' (`n.trials = 1`, `additive = FALSE`) are supported.  The activation is
#' detected automatically from the body of `m$Th`; the full set of qrnn
#' activation functions (`sigmoid`, `logistic`, `relu`, `lrelu`, `elu`,
#' `softplus`, `linear`) is supported.
#'
#' @param x A `qrnn_fit` object (created with [as_qrnn_fit()]).
#' @param ... Not used.
#' @param prefix A single string naming the output column.  Defaults to
#'   `".pred"`.
#'
#' @returns An [orbital][orbital_class] object.
#'
#' @seealso [as_qrnn_fit()], [qrnn::qrnn.fit()]
#'
#' @export
orbital.qrnn_fit <- function(x, ..., prefix = ".pred") {
    exprs <- .orbital_qrnn_impl(x, prefix = prefix)
    res <- new_orbital_class(exprs)
    attr(res, "pred_names") <- prefix
    res
}

# Internal implementation: derive orbital expressions from a qrnn_fit object.
.orbital_qrnn_impl <- function(x, prefix = ".pred") {
    weights <- x$weights

    if (!is.logical(x$additive) || isTRUE(x$additive)) {
        cli::cli_abort(
            "{.fn orbital} does not support additive qrnn models ({.code additive = TRUE})."
        )
    }

    W1 <- weights[[1L]]$W1 # (n_inputs + 1) x n_hidden  — last row = bias
    W2 <- weights[[1L]]$W2 # (n_hidden + 1) x n_outputs — last row = bias

    n_inputs <- nrow(W1) - 1L
    n_hidden <- ncol(W1)
    n_outputs <- ncol(W2)

    if (n_outputs != 1L) {
        cli::cli_abort(
            "{.fn orbital} only supports single-output qrnn models (n_outputs = 1)."
        )
    }

    input_names <- names(x$x.center)
    if (is.null(input_names) || length(input_names) != n_inputs) {
        input_names <- paste0("x", seq_len(n_inputs))
    }

    # -- Input standardisation: z_i = (x_i - center_i) / scale_i --------------
    norm_names <- paste0("orbital_qrnn_norm_", seq_len(n_inputs))
    norm_exprs <- vapply(
        seq_len(n_inputs),
        function(i) {
            glue::glue(
                "({backtick(input_names[i])} - {format_numeric(x$x.center[i])}) / {format_numeric(x$x.scale[i])}"
            )
        },
        character(1L)
    )

    # -- Hidden layer -----------------------------------------------------------
    # W1 layout: rows 1..n_inputs = weight rows, row n_inputs+1 = bias row
    weight_rows_h <- W1[seq_len(n_inputs), , drop = FALSE] # n_inputs x n_hidden
    bias_h <- W1[n_inputs + 1L, ] # length n_hidden

    # build_mlp_pre_act expects weight_mat (n_out x n_in) and biases (length n_out)
    hidden_pre_act <- build_mlp_pre_act(t(weight_rows_h), bias_h, norm_names)

    act_name <- .qrnn_activation_name(x$Th)
    hidden_act <- vapply(
        hidden_pre_act,
        function(z) {
            .qrnn_activation_expr(act_name, z)
        },
        character(1L)
    )

    hidden_names <- paste0("orbital_qrnn_h", seq_len(n_hidden))

    # -- Output layer -----------------------------------------------------------
    # W2 layout: rows 1..n_hidden = weight rows, row n_hidden+1 = bias row
    weight_rows_o <- W2[seq_len(n_hidden), , drop = FALSE] # n_hidden x 1
    bias_o <- W2[n_hidden + 1L, ] # length 1

    output_pre_act <- build_mlp_pre_act(t(weight_rows_o), bias_o, hidden_names)

    # De-standardise: y_hat = pre_act * y.scale + y.center
    out_expr <- glue::glue(
        "({output_pre_act[1L]}) * {format_numeric(x$y.scale[1L])} + {format_numeric(x$y.center[1L])}"
    )

    c(
        stats::setNames(norm_exprs, norm_names),
        stats::setNames(hidden_act, hidden_names),
        stats::setNames(as.character(out_expr), prefix)
    )
}

# Detect the activation function name from the body of m$Th.
.qrnn_activation_name <- function(Th_fn) {
    body_str <- paste(deparse(body(Th_fn)), collapse = " ")
    if (grepl("tanh(0.5", body_str, fixed = TRUE)) {
        "qrnn_sigmoid" # qrnn::sigmoid = tanh(0.5*x), NOT the logistic function
    } else if (
        grepl("0.5 + 0.5 * tanh", body_str, fixed = TRUE) ||
            grepl("0.5+0.5*tanh", body_str, fixed = TRUE)
    ) {
        "sigmoid" # qrnn::logistic = standard logistic/sigmoid
    } else if (grepl("ifelse(x >= 0, x, 0.01", body_str, fixed = TRUE)) {
        "leaky_relu"
    } else if (grepl("alpha * (exp(x)", body_str, fixed = TRUE)) {
        "elu"
    } else if (grepl("ifelse(x >= 0, x, 0)", body_str, fixed = TRUE)) {
        "relu"
    } else if (
        grepl("log(1 + exp(", body_str, fixed = TRUE) ||
            grepl("log1p(exp(", body_str, fixed = TRUE)
    ) {
        "softplus"
    } else if (grepl("^\\{\\s*x\\s*\\}$", trimws(body_str))) {
        "linear"
    } else {
        cli::cli_warn(
            c(
                "Could not identify qrnn activation from body of {.code m$Th}.",
                "i" = "Body: {body_str}",
                "i" = "Falling back to {.val linear}."
            ),
            class = "orbital_qrnn_unknown_activation"
        )
        "linear"
    }
}

# Produce a dplyr/SQL-portable expression for the detected qrnn activation.
.qrnn_activation_expr <- function(act_name, x_expr) {
    switch(
        act_name,
        # qrnn::sigmoid = tanh(0.5 * x) — note: this is NOT the logistic function
        "qrnn_sigmoid" = glue::glue("tanh(0.5 * ({x_expr}))"),
        "sigmoid" = glue::glue("1 / (1 + exp(-({x_expr})))"),
        "relu" = glue::glue("dplyr::if_else({x_expr} >= 0, {x_expr}, 0)"),
        "leaky_relu" = glue::glue(
            "dplyr::if_else({x_expr} >= 0, {x_expr}, 0.01 * ({x_expr}))"
        ),
        "elu" = glue::glue(
            "dplyr::if_else({x_expr} >= 0, {x_expr}, 1 * (exp({x_expr}) - 1))"
        ),
        "softplus" = glue::glue("log(1 + exp({x_expr}))"),
        "linear" = as.character(x_expr),
        cli::cli_abort("Unknown qrnn activation: {.val {act_name}}.")
    )
}
