# orbital method for H2O DeepLearning models (agua package)

orbital_h2o_dl_impl <- function(x, mode, type, lvl, prefix) {
  # Count weight matrices by iterating until h2o.weights() errors
  n_matrices <- 0L
  repeat {
    res <- tryCatch(
      h2o::h2o.weights(x, matrix_id = n_matrices + 1L),
      error = function(e) NULL
    )
    if (is.null(res)) {
      break
    }
    n_matrices <- n_matrices + 1L
  }

  # Activation for all hidden layers (dropout variants behave identically at inference)
  activation <- x@parameters$activation
  if (is.null(activation)) {
    activation <- "Rectifier"
  }

  # Maxout uses a grouped max-pooling architecture.
  # H2O stores `n_hidden * maxout_size` weight rows per layer; each neuron
  # corresponds to `maxout_size` sub-units and the output is their maximum.
  is_maxout <- activation %in% c("Maxout", "MaxoutWithDropout")
  maxout_size <- if (is_maxout) {
    sz <- x@parameters$maxout_size
    if (is.null(sz) || !is.numeric(sz) || sz < 2L) {
      cli::cli_abort(c(
        "Cannot determine {.arg maxout_size} for H2O Maxout model.",
        "i" = "{.code x@parameters$maxout_size} is missing or < 2."
      ))
    }
    as.integer(sz)
  } else {
    NULL
  }

  # These are NULL when standardize = FALSE was used during training
  norm_sub <- x@model$input_norm_sub
  norm_mul <- x@model$input_norm_mul

  all_exprs <- list()
  current_names <- NULL
  out_pre_act <- NULL

  for (mat_id in seq_len(n_matrices)) {
    wt_df <- as.data.frame(h2o::h2o.weights(x, matrix_id = mat_id))
    if (!identical(colnames(wt_df)[1L], "Bias")) {
      cli::cli_abort(c(
        "H2O weight matrix {.val {mat_id}} has unexpected first column {.val {colnames(wt_df)[1L]}}.",
        "i" = "orbital expects the bias column to be named {.val Bias} in {.fn h2o.weights} output."
      ))
    }
    biases <- wt_df[, 1L]
    wt_mat <- as.matrix(wt_df[, -1L, drop = FALSE])

    if (mat_id == 1L) {
      raw_input_names <- colnames(wt_mat)

      if (!is.null(norm_sub) && !is.null(norm_mul)) {
        # Emit normalization expressions: (x - mean) * (1/sd)
        norm_names <- paste0(
          "orbital_h2o_norm_",
          seq_along(raw_input_names)
        )
        norm_exprs <- vapply(
          seq_along(raw_input_names),
          function(i) {
            sub_i <- format_numeric(norm_sub[i])
            mul_i <- format_numeric(norm_mul[i])
            as.character(glue::glue(
              "({backtick(raw_input_names[i])} - {sub_i}) * {mul_i}"
            ))
          },
          character(1)
        )
        all_exprs[["norm"]] <- stats::setNames(norm_exprs, norm_names)
        current_names <- norm_names
      } else {
        current_names <- raw_input_names
      }
    }

    pre_act <- build_mlp_pre_act(wt_mat, biases, current_names)

    if (mat_id < n_matrices) {
      # Hidden layer: apply activation
      if (is_maxout) {
        # Maxout: each neuron = max over maxout_size consecutive sub-units.
        # H2O stores rows in groups of maxout_size: [sub1_n1, sub2_n1, ..., subK_n1, sub1_n2, ...]
        n_sub_units <- nrow(wt_mat)
        n_neurons <- n_sub_units %/% maxout_size
        if (n_sub_units %% maxout_size != 0L) {
          cli::cli_abort(c(
            "H2O Maxout layer {mat_id}: weight matrix has {n_sub_units} rows which",
            "is not divisible by maxout_size={maxout_size}.",
            "i" = "Expected {maxout_size} sub-units per neuron."
          ))
        }
        act <- vapply(
          seq_len(n_neurons),
          function(neuron) {
            sub_exprs <- pre_act[
              ((neuron - 1L) * maxout_size + 1L):(neuron * maxout_size)
            ]
            # Build pmax(sub1, sub2, ..., subK)
            paste0("pmax(", paste(sub_exprs, collapse = ", "), ")")
          },
          character(1)
        )
        layer_names <- paste0("orbital_mlp_l", mat_id, "_h", seq_len(n_neurons))
      } else {
        act <- vapply(
          pre_act,
          function(z) activation_expr(activation, z),
          character(1)
        )
        layer_names <- paste0(
          "orbital_mlp_l",
          mat_id,
          "_h",
          seq_len(nrow(wt_mat))
        )
      }
      all_exprs[[paste0("l", mat_id)]] <- stats::setNames(
        act,
        layer_names
      )
      current_names <- layer_names
    } else {
      # Output layer: keep raw pre-activations
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

#' @export
orbital.H2ODeepLearningModel <- function(
  x,
  ...,
  mode = c("regression", "classification"),
  type = NULL,
  lvl = NULL,
  prefix = ".pred"
) {
  mode <- rlang::arg_match(mode)
  type <- default_type(type)
  orbital_h2o_dl_impl(
    x,
    mode = mode,
    type = type,
    lvl = lvl,
    prefix = prefix
  )
}
