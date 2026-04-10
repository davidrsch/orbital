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

  if (!inherits(x, "nnet")) {
    cli::cli_abort(c(
      "{.fn orbital} for the {.pkg nnet} backend requires a single {.cls nnet} object.",
      "i" = "Bagged models ({.code bag_mlp(engine = \"nnet\")}) are not yet supported."
    ))
  }

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
    # Binary classification.
    # parsnip post-processes nnet raw output p = sigmoid(linear) via:
    #   cbind(1-p, p)  then  row-wise softmax
    # giving: pred_0 = exp(1-p)/(exp(1-p)+exp(p)),
    #         pred_1 = exp(p)/(exp(1-p)+exp(p))
    sigmoid_expr <- activation_expr("sigmoid", output_pre_act[1L])

    c(hidden_exprs, binary_from_prob(sigmoid_expr, type, lvl))
  } else {
    # Multiclass classification.
    # nnet output layer applies softmax internally, so output_pre_act passed
    # through one softmax gives the true probability distribution.
    lvl_bt <- backtick(lvl)
    norm1_col <- "orbital_nnet_norm1"
    norm1_bt <- backtick(norm1_col)
    raw_cols <- paste0("orbital_nnet_raw_", seq_along(lvl))
    raw_bt <- backtick(raw_cols)

    logit_exprs <- stats::setNames(output_pre_act, lvl)
    norm1_expr <- glue::glue_collapse(
      glue::glue("exp({lvl_bt})"),
      sep = " + "
    )
    raw_exprs <- stats::setNames(
      glue::glue("exp({lvl_bt}) / {norm1_bt}"),
      raw_cols
    )

    res <- c(
      logit_exprs,
      stats::setNames(norm1_expr, norm1_col),
      raw_exprs
    )

    if ("class" %in% type) {
      # argmax is invariant to monotone transforms, use logit columns
      res <- c(res, orbital_tmp_class_name = softmax_class(lvl))
    }
    if ("prob" %in% type) {
      prob_exprs <- as.character(raw_bt)
      names(prob_exprs) <- paste0("orbital_tmp_prob_name", seq_along(lvl))
      res <- c(res, prob_exprs)
    }
    c(hidden_exprs, res)
  }
}
