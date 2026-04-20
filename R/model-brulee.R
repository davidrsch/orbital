# orbital methods for brulee MLP models

# Try to extract per-layer activation alpha values from the brulee torch model.
# Returns a numeric vector of length n_h_layers, or NULL when the brulee object
# does not carry a live torch model (e.g. reloaded without torch available).
# Errors while probing an actual live model are surfaced via cli::cli_warn
# rather than swallowed, so callers know the alpha fallback is in use.
brulee_extract_alphas <- function(x, activations, n_h_layers) {
  if (is.null(x$fit) || is.null(x$fit$model)) {
    return(NULL)
  }
  modules <- tryCatch(
    x$fit$model$named_modules(),
    error = function(e) {
      cli::cli_warn(c(
        "brulee MLP: could not enumerate torch modules ({conditionMessage(e)}).",
        i = "Falling back to ONNX / parsnip default activation alphas."
      ))
      NULL
    }
  )
  if (is.null(modules)) {
    return(NULL)
  }
  # Collect activation modules in layer order (act1, act2, ...)
  act_keys <- paste0("act", seq_len(n_h_layers))
  alphas <- vector("list", n_h_layers)
  for (i in seq_len(n_h_layers)) {
    mod <- modules[[act_keys[i]]]
    if (is.null(mod)) {
      next
    }
    act <- activations[i]
    if (act %in% c("leaky_relu")) {
      ns <- tryCatch(
        as.numeric(mod$negative_slope),
        error = function(e) NULL
      )
      alphas[[i]] <- ns
    } else if (act %in% c("elu", "celu")) {
      a <- tryCatch(
        as.numeric(mod$alpha),
        error = function(e) NULL
      )
      alphas[[i]] <- a
    } else if (act == "prelu") {
      # PReLU has per-channel learnable weights; brulee exports them via
      # coef(x) under a key like "act<i>.weight" or "prelu.weight".
      # The full weight vector is returned so each neuron gets its own alpha.
      coef_list <- tryCatch(stats::coef(x), error = function(e) NULL)
      prelu_val <- NULL
      candidate_keys <- c(
        paste0("act", i, ".weight"),
        "prelu.weight",
        "weight"
      )
      if (!is.null(coef_list)) {
        # Try common key patterns brulee may use for PReLU weights
        for (k in candidate_keys) {
          if (!is.null(coef_list[[k]])) {
            prelu_val <- tryCatch(
              as.numeric(coef_list[[k]]),
              error = function(e) NULL
            )
            if (!is.null(prelu_val)) break
          }
        }
      }
      if (is.null(prelu_val)) {
        cli::cli_abort(
          c(
            "PReLU weight not found in {.code coef()} output for layer {i}.",
            "i" = "orbital cannot determine the learned per-channel slopes.",
            "i" = "Tried keys: {.val {paste(candidate_keys, collapse = ', ')}}.",
            "i" = "Known coef() keys: {.val {paste(names(coef_list %||% list()), collapse = ', ')}}.",
            "i" = "Please file an issue with your brulee version and {.code coef()} output."
          )
        )
      }
      alphas[[i]] <- prelu_val
    }
  }
  alphas
}

orbital_brulee_mlp_impl <- function(x, mode, type, lvl, prefix) {
  coef_obj <- stats::coef(x)
  input_names <- x$dims$features
  n_h_layers <- length(x$dims$h)

  # Build per-layer activation vector. brulee stores this in two shapes:
  #   * scalar (default): applied to every hidden layer
  #   * length-n_h_layers vector: already per-layer
  # brulee_mlp_two_layer stores the second activation separately under
  # "activation_2"; preserve backward compatibility with that layout.
  activations <- local({
    a1 <- x$parameters$activation
    a2 <- x$parameters$activation_2
    if (!is.null(a2) && n_h_layers == 2L) {
      c(a1, a2)
    } else if (length(a1) == 1L) {
      rep_len(a1, n_h_layers)
    } else if (length(a1) == n_h_layers) {
      a1
    } else {
      cli::cli_abort(c(
        "brulee activation parameter has unexpected length.",
        "i" = "Got length {length(a1)}; expected 1 or {n_h_layers}."
      ))
    }
  })

  alphas <- brulee_extract_alphas(x, activations, n_h_layers)

  all_exprs <- list()
  current_names <- input_names

  brulee_require_coef <- function(key, layer_label) {
    val <- coef_obj[[key]]
    if (is.null(val)) {
      cli::cli_abort(c(
        "brulee fit is missing {.code {key}} required for layer {layer_label}.",
        "i" = "The fit object may be corrupted or produced by an \
               incompatible brulee version.",
        "i" = "Available keys: {.val {names(coef_obj)}}."
      ))
    }
    val
  }

  for (i in seq_len(n_h_layers)) {
    w <- brulee_require_coef(paste0("fc", i, ".weight"), paste0("hidden ", i))
    b <- brulee_require_coef(paste0("fc", i, ".bias"), paste0("hidden ", i))
    if (NROW(w) != length(b)) {
      cli::cli_abort(c(
        "brulee hidden layer {i}: weight/bias shape mismatch.",
        i = "fc{i}.weight has {NROW(w)} rows but fc{i}.bias has length {length(b)}.",
        i = "The brulee fit object may be corrupted or produced by an incompatible version."
      ))
    }
    pre_act <- build_mlp_pre_act(w, b, current_names)
    layer_names <- paste0("orbital_mlp_l", i, "_h", seq_len(nrow(w)))
    alpha_i <- if (!is.null(alphas)) alphas[[i]] else NULL
    if (activations[i] == "softmax") {
      norm_col <- paste0("orbital_mlp_l", i, "_softmax_norm")
      all_exprs[[i]] <- softmax_hidden_exprs(pre_act, layer_names, norm_col)
    } else if (activations[i] == "prelu" && length(alpha_i) > 1L) {
      act <- vapply(
        seq_along(pre_act),
        function(j) {
          activation_expr(activations[i], pre_act[[j]], alpha = alpha_i[[j]])
        },
        character(1)
      )
      all_exprs[[i]] <- stats::setNames(act, layer_names)
    } else {
      act <- vapply(
        pre_act,
        function(z) activation_expr(activations[i], z, alpha = alpha_i),
        character(1)
      )
      all_exprs[[i]] <- stats::setNames(act, layer_names)
    }
    current_names <- layer_names
  }

  fc_out <- n_h_layers + 1L
  w_out <- brulee_require_coef(paste0("fc", fc_out, ".weight"), "output")
  b_out <- brulee_require_coef(paste0("fc", fc_out, ".bias"), "output")
  if (NROW(w_out) != length(b_out)) {
    cli::cli_abort(c(
      "brulee output layer: weight/bias shape mismatch.",
      i = "fc{fc_out}.weight has {NROW(w_out)} rows but fc{fc_out}.bias has length {length(b_out)}.",
      i = "The brulee fit object may be corrupted or produced by an incompatible version."
    ))
  }
  out_pre_act <- build_mlp_pre_act(w_out, b_out, current_names)

  hidden_exprs <- unlist(all_exprs, use.names = TRUE)

  n_out <- x$dims$y

  if (mode == "regression") {
    y_mean <- format_numeric(x$y_stats$mean)
    y_sd <- format_numeric(x$y_stats$sd)
    final_expr <- glue::glue("({out_pre_act[1]}) * {y_sd} + {y_mean}")
    res <- c(
      hidden_exprs,
      stats::setNames(as.character(final_expr), prefix)
    )
  } else if (n_out == 1L) {
    sigmoid_expr <- activation_expr("sigmoid", out_pre_act[1])
    res <- c(hidden_exprs, binary_from_prob(sigmoid_expr, type, lvl))
  } else {
    logit_exprs <- stats::setNames(out_pre_act, lvl)
    res <- c(hidden_exprs, multiclass_from_logits(logit_exprs, type, lvl))
  }

  res
}

#' @rdname orbital
#' @method orbital brulee_mlp
#' @section brulee_mlp backend:
#'   Supported activations (from `brulee::brulee_activations()`): `celu`,
#'   `elu`, `gelu`, `hardshrink`, `hardsigmoid`, `hardtanh`, `leaky_relu`,
#'   `linear`, `log_sigmoid`, `relu`, `relu6`, `rrelu`, `selu`, `sigmoid`,
#'   `silu`, `softplus`, `softshrink`, `softsign`, `tanh`, `tanhshrink`.
#'
#'   Deep brulee MLPs with `n_h_layers > 2` are supported as of this release;
#'   per-layer activations use the `activation` argument as a vector or scalar
#'   (recycled).
#'
#'   Activation alphas (`leaky_relu`, `elu`, `celu`, `prelu`) are extracted
#'   from the live torch module when available, with a warning-only fallback
#'   to parsnip / ONNX defaults when the torch model cannot be probed. PReLU
#'   per-channel weights are honoured when exported via [stats::coef()].
#' @export
orbital.brulee_mlp <- function(
  x,
  ...,
  mode = c("regression", "classification"),
  type = NULL,
  lvl = NULL,
  prefix = ".pred"
) {
  mode <- rlang::arg_match(mode)
  type <- default_type(type)
  orbital_brulee_mlp_impl(
    x,
    mode = mode,
    type = type,
    lvl = lvl,
    prefix = prefix
  )
}

#' @rdname orbital
#' @method orbital brulee_mlp_two_layer
#' @section brulee_mlp_two_layer backend:
#'   Convenience wrapper identical in behaviour to [orbital.brulee_mlp()] but
#'   for models fitted via [brulee::brulee_mlp_two_layer()], which stores the
#'   second hidden activation under the `activation_2` parameter slot.
#' @export
orbital.brulee_mlp_two_layer <- function(
  x,
  ...,
  mode = c("regression", "classification"),
  type = NULL,
  lvl = NULL,
  prefix = ".pred"
) {
  mode <- rlang::arg_match(mode)
  type <- default_type(type)
  orbital_brulee_mlp_impl(
    x,
    mode = mode,
    type = type,
    lvl = lvl,
    prefix = prefix
  )
}
