# orbital methods for brulee MLP models

# Try to extract per-layer activation alpha values from the brulee torch model.
# Returns a numeric vector of length n_h_layers, or NULL if unavailable.
brulee_extract_alphas <- function(x, activations, n_h_layers) {
  tryCatch(
    {
      if (is.null(x$fit) || is.null(x$fit$model)) {
        return(NULL)
      }
      modules <- x$fit$model$named_modules()
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
          # Since activation_expr() only accepts a scalar alpha, we reduce to
          # the first element with a warning.
          coef_list <- tryCatch(stats::coef(x), error = function(e) NULL)
          prelu_val <- NULL
          if (!is.null(coef_list)) {
            # Try common key patterns brulee may use for PReLU weights
            candidate_keys <- c(
              paste0("act", i, ".weight"),
              "prelu.weight",
              "weight"
            )
            for (k in candidate_keys) {
              if (!is.null(coef_list[[k]])) {
                prelu_val <- tryCatch(
                  as.numeric(coef_list[[k]])[1],
                  error = function(e) NULL
                )
                if (!is.null(prelu_val)) break
              }
            }
          }
          if (is.null(prelu_val)) {
            cli::cli_warn(
              c(
                "PReLU weight not found in {.code coef()} output for layer {i}.",
                "i" = "Falling back to PyTorch default init alpha = 0.25.",
                "i" = "Full per-channel PReLU support requires an architectural change."
              )
            )
            prelu_val <- 0.25
          } else {
            cli::cli_warn(
              c(
                "PReLU uses per-channel learnable weights, but {.fn activation_expr} only accepts a scalar alpha.",
                "i" = "Reducing layer {i} PReLU weights to the first channel value ({prelu_val}).",
                "i" = "Full per-channel support requires an architectural change."
              )
            )
          }
          alphas[[i]] <- prelu_val
        }
      }
      alphas
    },
    error = function(e) NULL
  )
}

orbital_brulee_mlp_impl <- function(x, mode, type, lvl, prefix) {
  coef_obj <- stats::coef(x)
  input_names <- x$dims$features
  activations <- x$parameters$activation
  n_h_layers <- length(x$dims$h)

  alphas <- brulee_extract_alphas(x, activations, n_h_layers)

  all_exprs <- list()
  current_names <- input_names

  for (i in seq_len(n_h_layers)) {
    w <- coef_obj[[paste0("fc", i, ".weight")]]
    b <- coef_obj[[paste0("fc", i, ".bias")]]
    pre_act <- build_mlp_pre_act(w, b, current_names)
    alpha_i <- if (!is.null(alphas)) alphas[[i]] else NULL
    act <- vapply(
      pre_act,
      function(z) activation_expr(activations[i], z, alpha = alpha_i),
      character(1)
    )
    layer_names <- paste0("orbital_mlp_l", i, "_h", seq_len(nrow(w)))
    all_exprs[[i]] <- stats::setNames(act, layer_names)
    current_names <- layer_names
  }

  fc_out <- n_h_layers + 1L
  w_out <- coef_obj[[paste0("fc", fc_out, ".weight")]]
  b_out <- coef_obj[[paste0("fc", fc_out, ".bias")]]
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
