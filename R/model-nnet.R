#' @rdname orbital
#' @method orbital nnet
#' @section nnet backend:
#'   Supports single-hidden-layer MLPs fitted with `skip = FALSE`. orbital
#'   refuses models with unsupported output-activation combinations:
#'   regression requires `linout = TRUE`; classification requires
#'   `linout = FALSE`, `softmax = FALSE` for binary (1 output) and
#'   `softmax = TRUE` for multi-class. `censored = TRUE` is not supported.
#'   `x$entropy` (loss-function flag) is intentionally ignored because it does
#'   not affect inference-time outputs.
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

  if (isTRUE(x$skip)) {
    cli::cli_abort(c(
      "{.fn orbital} does not support {.cls nnet} models fitted with {.code skip = TRUE}.",
      "i" = "Skip connections change the weight vector layout in an unsupported way.",
      "i" = "Re-fit the model with {.code skip = FALSE}."
    ))
  }

  # ----------------------------------------------------------------------------
  # Output activation validation.
  #
  # nnet applies different output transforms depending on `linout`, `softmax`,
  # and `censored` (see ?nnet::nnet). orbital's expression builder assumes
  # specific defaults per-mode; other combinations would silently produce
  # wrong predictions, so we refuse them up front.
  #
  # Supported combinations:
  #   mode = "regression"     : linout = TRUE, softmax = FALSE  (raw linear output)
  #   mode = "classification" : linout = FALSE, softmax = FALSE for n_out == 1
  #                             (internal sigmoid, parsnip softmaxes to 2 classes)
  #   mode = "classification" : linout = FALSE, softmax = TRUE  for n_out > 1
  #                             (internal softmax, parsnip re-softmaxes)
  # Anything else is refused rather than silently miscompiled.
  # ----------------------------------------------------------------------------
  linout <- isTRUE(x$linout)
  softmax_flag <- isTRUE(x$softmax)
  censored <- isTRUE(x$censored)
  entropy <- isTRUE(x$entropy)

  if (censored) {
    cli::cli_abort(c(
      "{.fn orbital} does not support {.cls nnet} models fitted with {.code censored = TRUE}.",
      "i" = "The censored multinomial output transform is not implemented."
    ))
  }

  if (mode == "regression") {
    if (!linout) {
      cli::cli_abort(c(
        "{.fn orbital} for {.pkg nnet} regression requires {.code linout = TRUE}.",
        "x" = "Got {.code linout = FALSE}; nnet would internally apply a \
               logistic squash to the output, so emitting the raw pre-activation \
               would produce silently wrong predictions.",
        "i" = "Re-fit with {.code nnet::nnet(..., linout = TRUE)} (parsnip / \
               {.code mlp(mode = \"regression\", engine = \"nnet\")} does this \
               automatically)."
      ))
    }
    if (softmax_flag) {
      cli::cli_abort(c(
        "{.fn orbital} for {.pkg nnet} regression does not support \
         {.code softmax = TRUE} (got {.code softmax = TRUE}).",
        "i" = "Re-fit with {.code softmax = FALSE}."
      ))
    }
  } else if (mode == "classification") {
    if (linout) {
      cli::cli_abort(c(
        "{.fn orbital} for {.pkg nnet} classification requires {.code linout = FALSE}.",
        "x" = "Got {.code linout = TRUE}; with a linear output nnet would not \
               apply the sigmoid / softmax normalisation that parsnip expects."
      ))
    }
    n_out_tmp <- x$n[3L]
    if (n_out_tmp == 1L && softmax_flag) {
      cli::cli_abort(c(
        "{.fn orbital} for binary {.pkg nnet} classification (n_out = 1) \
         expects {.code softmax = FALSE} (nnet emits a single sigmoid unit).",
        "x" = "Got {.code softmax = TRUE}."
      ))
    }
    if (n_out_tmp > 1L && !softmax_flag) {
      cli::cli_abort(c(
        "{.fn orbital} for multi-class {.pkg nnet} classification (n_out = {n_out_tmp}) \
         expects {.code softmax = TRUE}.",
        "x" = "Got {.code softmax = FALSE}; outputs would be independent logistic units \
              rather than a softmax simplex."
      ))
    }
  }

  # x$entropy toggles the loss function during fitting (cross-entropy vs SSE);
  # it does not change the inference-time output transform, so orbital can
  # safely ignore it. Recorded here for transparency.
  invisible(entropy)

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
    # nnet emits a single sigmoid probability p for the second level. parsnip's
    # probability path then softmaxes the two-column expansion cbind(1 - p, p),
    # so orbital mirrors that exact post-processing for numerical parity.
    sigmoid_expr <- activation_expr("sigmoid", output_pre_act[1L])
    res <- NULL

    if ("class" %in% type) {
      res <- c(res, binary_from_prob(sigmoid_expr, "class", lvl))
    }
    if ("prob" %in% type) {
      # Confirmed correct: parsnip::nnet_softmax (mlp.R) applies softmax to cbind(1-p, p) for binary.
      neg_expr <- glue::glue("1 - ({sigmoid_expr})")
      denom_expr <- glue::glue("exp({neg_expr}) + exp({sigmoid_expr})")
      res <- c(
        res,
        orbital_tmp_prob_name1 = glue::glue("exp({neg_expr}) / ({denom_expr})"),
        orbital_tmp_prob_name2 = glue::glue(
          "exp({sigmoid_expr}) / ({denom_expr})"
        )
      )
    }

    c(hidden_exprs, res)
  } else {
    # Multiclass classification.
    # nnet's raw multiclass outputs are post-softmax probabilities, and parsnip
    # applies one more softmax-style normalization step when returning
    # class probabilities. Mirror that behavior for exact predict() parity.
    logit_cols <- paste0("orbital_nnet_logit_", seq_along(lvl))
    logit_bt <- backtick(logit_cols)
    norm1_col <- "orbital_nnet_norm1"
    norm1_bt <- backtick(norm1_col)
    raw_cols <- paste0("orbital_nnet_raw_", seq_along(lvl))
    raw_bt <- backtick(raw_cols)
    norm2_col <- "orbital_nnet_norm2"
    norm2_bt <- backtick(norm2_col)

    logit_exprs <- stats::setNames(output_pre_act, logit_cols)
    norm1_expr <- glue::glue_collapse(
      glue::glue("exp({logit_bt})"),
      sep = " + "
    )
    raw_exprs <- stats::setNames(
      glue::glue("exp({logit_bt}) / {norm1_bt}"),
      raw_cols
    )

    res <- c(
      logit_exprs,
      stats::setNames(norm1_expr, norm1_col),
      raw_exprs
    )

    if ("class" %in% type) {
      # argmax is invariant to monotone transforms, so compare the logit columns
      # while still returning the original class labels from `lvl`.
      res <- c(res, orbital_tmp_class_name = softmax_class(logit_cols, lvl))
    }
    if ("prob" %in% type) {
      # Confirmed correct: parsnip::nnet_softmax (mlp.R) re-normalises multiclass output with softmax (double-softmax is intentional).
      norm2_expr <- glue::glue_collapse(
        glue::glue("exp({raw_bt})"),
        sep = " + "
      )
      prob_exprs <- stats::setNames(
        glue::glue("exp({raw_bt}) / {norm2_bt}"),
        paste0("orbital_tmp_prob_name", seq_along(lvl))
      )
      res <- c(res, stats::setNames(norm2_expr, norm2_col), prob_exprs)
    }
    c(hidden_exprs, res)
  }
}
