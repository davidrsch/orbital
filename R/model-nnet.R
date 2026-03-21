#' @export
orbital.nnet <- function(
    x,
    ...,
    mode = c("regression", "classification"),
    type = NULL,
    lvl = NULL,
    prefix = ".pred"
) {
    mode <- rlang::arg_match(mode)
    type <- default_type(type)

    n_in <- x$n[1L]
    n_h <- x$n[2L]
    n_out <- x$n[3L]

    input_names <- x$coefnames

    # Build hidden layer weight matrix (n_h x n_in) and bias vector
    hidden_w <- matrix(0, nrow = n_h, ncol = n_in)
    hidden_b <- numeric(n_h)
    for (i in seq_len(n_h)) {
        base <- (i - 1L) * (n_in + 1L)
        hidden_b[i] <- x$wts[base + 1L]
        for (j in seq_len(n_in)) {
            hidden_w[i, j] <- x$wts[base + j + 1L]
        }
    }

    # Hidden pre-activations and sigmoid activations
    hidden_pre_act <- build_mlp_pre_act(hidden_w, hidden_b, input_names)
    hidden_act <- vapply(
        hidden_pre_act,
        function(z) activation_expr("sigmoid", z),
        character(1)
    )
    hidden_names <- paste0("orbital_mlp_h", seq_len(n_h))
    hidden_exprs <- stats::setNames(hidden_act, hidden_names)

    # Build output layer weight matrix (n_out x n_h) and bias vector
    off <- n_h * (n_in + 1L)
    output_w <- matrix(0, nrow = n_out, ncol = n_h)
    output_b <- numeric(n_out)
    for (k in seq_len(n_out)) {
        base <- off + (k - 1L) * (n_h + 1L)
        output_b[k] <- x$wts[base + 1L]
        for (j in seq_len(n_h)) {
            output_w[k, j] <- x$wts[base + j + 1L]
        }
    }

    output_pre_act <- build_mlp_pre_act(output_w, output_b, hidden_names)

    if (mode == "regression") {
        c(hidden_exprs, stats::setNames(output_pre_act[1L], prefix))
    } else if (mode == "classification" && n_out == 1L) {
        sigmoid_expr <- activation_expr("sigmoid", output_pre_act[1L])
        c(hidden_exprs, binary_from_prob(sigmoid_expr, type, lvl))
    } else {
        c(
            hidden_exprs,
            multiclass_from_logits(
                stats::setNames(output_pre_act, lvl),
                type,
                lvl
            )
        )
    }
}
