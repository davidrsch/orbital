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
    if (is.null(sz) || !is.numeric(sz) || length(sz) != 1L || sz < 2L) {
      cli::cli_abort(c(
        "Could not resolve {.arg maxout_size} from the H2O model.",
        "x" = "orbital will not silently fall back to a default because an \
               incorrect {.arg maxout_size} silently mispredicts.",
        "i" = "Refit the model explicitly setting {.arg maxout_size}, or file \
               an issue with the H2O version and {.code x@parameters} dump."
      ))
    }
    if (sz > .Machine$integer.max || sz != floor(sz)) {
      cli::cli_abort(c(
        "H2O {.arg maxout_size} = {.val {sz}} is not a representable positive integer.",
        i = "Expected a small positive integer (typically 2-5)."
      ))
    }
    as.integer(sz)
  } else {
    NULL
  }

  # These are NULL when standardize = FALSE was used during training
  norm_sub <- x@model$input_norm_sub
  norm_mul <- x@model$input_norm_mul
  if (xor(is.null(norm_sub), is.null(norm_mul))) {
    cli::cli_abort(
      "H2O model normalisation vectors are inconsistent; the model may be corrupted."
    )
  }
  if (!is.null(norm_sub) && (!is.numeric(norm_sub) || anyNA(norm_sub))) {
    cli::cli_abort(
      "H2O {.code input_norm_sub} is not a finite numeric vector; the model may be corrupted."
    )
  }
  if (!is.null(norm_mul) && (!is.numeric(norm_mul) || anyNA(norm_mul))) {
    cli::cli_abort(
      "H2O {.code input_norm_mul} is not a finite numeric vector; the model may be corrupted."
    )
  }

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
        if (
          length(norm_sub) != length(raw_input_names) ||
            length(norm_mul) != length(raw_input_names)
        ) {
          cli::cli_abort(c(
            "H2O normalisation vectors have unexpected length.",
            "i" = paste0(
              "norm_sub has length ",
              length(norm_sub),
              ", ",
              "norm_mul has length ",
              length(norm_mul),
              ", but ",
              "the weight matrix has ",
              length(raw_input_names),
              " input columns."
            )
          ))
        }
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
    if (length(out_pre_act) > 1L) {
      cli::cli_abort(c(
        "H2O multi-output regression is not yet supported by orbital.",
        "i" = paste0("The model has ", length(out_pre_act), " output neurons."),
        "i" = "Only single-output regression models are currently supported."
      ))
    }
    c(hidden_exprs, stats::setNames(out_pre_act[1L], prefix))
  } else if (n_out == 1L) {
    sigmoid_expr <- activation_expr("sigmoid", out_pre_act[1L])
    c(hidden_exprs, binary_from_prob(sigmoid_expr, type, lvl))
  } else {
    if (length(lvl) != n_out) {
      cli::cli_abort(c(
        "H2O model output size ({n_out}) does not match the number of classes ({length(lvl)}).",
        "i" = "lvl must be in H2O alphabetical class order."
      ))
    }
    h2o_levels <- h2o_response_levels(x)
    if (is.null(h2o_levels)) {
      cli::cli_warn(c(
        "H2O class-order verification skipped: response levels could not be \
         extracted from the model object.",
        i = "orbital will trust {.arg lvl} = {.val {lvl}} to be in H2O's \
            alphabetical output order.",
        i = "If this ordering is wrong, multi-class probability columns will \
            be silently misaligned.",
        i = "File an issue with your H2O version and {.code x@model$output} \
            structure if you hit this."
      ))
    } else {
      if (!setequal(as.character(lvl), as.character(h2o_levels))) {
        cli::cli_abort(c(
          "Class labels in {.arg lvl} do not match the H2O model's response levels.",
          "i" = "{.arg lvl}: {.val {lvl}}",
          "i" = "H2O levels: {.val {h2o_levels}}"
        ))
      }
      if (!identical(as.character(lvl), as.character(h2o_levels))) {
        cli::cli_abort(c(
          "{.arg lvl} is not in the H2O model's class order.",
          "i" = "H2O output columns are ordered as {.val {h2o_levels}}; pass \
                 {.arg lvl} in that exact order to avoid silent misalignment."
        ))
      }
    }
    logit_exprs <- stats::setNames(out_pre_act, lvl)
    c(hidden_exprs, multiclass_from_logits(logit_exprs, type, lvl))
  }
}

# Extract the response-factor levels from an H2O DeepLearning model in the
# exact order that H2O uses for its probability columns. Returns NULL when
# the levels cannot be determined (e.g. older H2O model layouts); callers
# must treat NULL as "unable to verify" and not "verified".
h2o_response_levels <- function(x) {
  tryCatch(
    {
      # H2O stores per-column domains in @model$output$domains aligned with
      # @model$output$names. The response column's domain contains the
      # class labels in H2O's alphabetical output order.
      response_col <- x@parameters$response_column
      if (is.null(response_col)) {
        response_col <- x@allparameters$response_column
      }
      if (is.null(response_col)) {
        return(NULL)
      }
      names_vec <- x@model$output$names
      domains <- x@model$output$domains
      if (is.null(names_vec) || is.null(domains)) {
        return(NULL)
      }
      idx <- match(response_col, names_vec)
      if (is.na(idx)) {
        return(NULL)
      }
      lev <- domains[[idx]]
      if (is.null(lev) || !is.character(lev) || !length(lev)) {
        return(NULL)
      }
      lev
    },
    error = function(e) NULL
  )
}

#' @rdname orbital
#' @method orbital H2ODeepLearningModel
#' @section H2O DeepLearning backend:
#'   Supports hidden activations `"Rectifier"` / `"RectifierWithDropout"`,
#'   `"Tanh"` / `"TanhWithDropout"`, and `"Maxout"` / `"MaxoutWithDropout"`
#'   (requires `maxout_size` to be recoverable from the fit). `"Sigmoid"` is
#'   accepted via the generic activation table but is not a standard H2O
#'   choice; verify the fit path if your model reports it. Multi-output
#'   regression is explicitly rejected.
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
