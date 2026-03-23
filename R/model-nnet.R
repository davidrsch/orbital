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
    # Binary classification.
    # parsnip post-processes nnet raw output p = sigmoid(linear) via:
    #   cbind(1-p, p)  then  row-wise softmax
    # giving: pred_0 = exp(1-p)/(exp(1-p)+exp(p)),
    #         pred_1 = exp(p)/(exp(1-p)+exp(p))
    sigmoid_expr <- activation_expr("sigmoid", output_pre_act[1L])

    if ("prob" %in% type) {
      sig_col <- "orbital_nnet_sigmoid"
      norm_col <- "orbital_nnet_norm"
      sig_bt <- backtick(sig_col)
      norm_bt <- backtick(norm_col)

      intermediates <- c(
        stats::setNames(sigmoid_expr, sig_col),
        stats::setNames(
          glue::glue("exp(1 - {sig_bt}) + exp({sig_bt})"),
          norm_col
        )
      )

      res <- NULL
      if ("class" %in% type) {
        levels_q <- glue::double_quote(lvl)
        res <- c(
          res,
          orbital_tmp_class_name = as.character(glue::glue(
            "dplyr::case_when({sig_bt} > 0.5 ~ {levels_q[2]}, .default = {levels_q[1]})"
          ))
        )
      }
      res <- c(
        res,
        orbital_tmp_prob_name1 = as.character(
          glue::glue("exp(1 - {sig_bt}) / {norm_bt}")
        ),
        orbital_tmp_prob_name2 = "1 - `orbital_tmp_prob_name1`"
      )
      c(hidden_exprs, intermediates, res)
    } else {
      # Class-only: sigmoid threshold is equivalent to parsnip's class
      c(hidden_exprs, binary_from_prob(sigmoid_expr, type, lvl))
    }
  } else {
    # Multiclass classification.
    # parsnip post-processes nnet raw output (= softmax of linear logits)
    # by applying softmax again:
    #   1) nnet_raw_k  = exp(l_k) / sum_j exp(l_j)   [nnet internal softmax]
    #   2) pred_k      = exp(nnet_raw_k) / sum_j exp(nnet_raw_j)  [parsnip post]
    lvl_bt <- backtick(lvl)
    norm1_col <- "orbital_nnet_norm1"
    norm1_bt <- backtick(norm1_col)
    raw_cols <- paste0("orbital_nnet_raw_", seq_along(lvl))
    raw_bt <- backtick(raw_cols)
    norm2_col <- "orbital_nnet_norm2"
    norm2_bt <- backtick(norm2_col)

    logit_exprs <- stats::setNames(output_pre_act, lvl)
    norm1_expr <- glue::glue_collapse(
      glue::glue("exp({lvl_bt})"),
      sep = " + "
    )
    raw_exprs <- stats::setNames(
      glue::glue("exp({lvl_bt}) / {norm1_bt}"),
      raw_cols
    )
    norm2_expr <- glue::glue_collapse(
      glue::glue("exp({raw_bt})"),
      sep = " + "
    )

    res <- c(
      logit_exprs,
      stats::setNames(norm1_expr, norm1_col),
      raw_exprs,
      stats::setNames(norm2_expr, norm2_col)
    )

    if ("class" %in% type) {
      # argmax is invariant to monotone transforms, use logit columns
      res <- c(res, orbital_tmp_class_name = softmax_class(lvl))
    }
    if ("prob" %in% type) {
      prob_exprs <- glue::glue("exp({raw_bt}) / {norm2_bt}")
      names(prob_exprs) <- paste0("orbital_tmp_prob_name", seq_along(lvl))
      res <- c(res, prob_exprs)
    }
    c(hidden_exprs, res)
  }
}
