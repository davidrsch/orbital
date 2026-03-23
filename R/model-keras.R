# orbital methods for all keras-based models (legacy keras R package and keras3/kerasnip)

# Internal: parse inbound layer names from one layer's config entry.
# Handles keras3 new format ({args: [{keras_history: ["name", 0, 0]}]})
# and old keras format ([[["name", 0, 0], ...]]).
.keras_parse_inbound <- function(layer_cfg) {
  nodes <- layer_cfg[["inbound_nodes"]]
  if (is.null(nodes) || length(nodes) == 0L) {
    return(character(0L))
  }

  first_node <- nodes[[1L]]

  # keras3 format: first_node = list(args = list(...), kwargs = list())
  args <- first_node[["args"]]
  if (!is.null(args) && is.list(args)) {
    res <- vapply(
      args,
      function(arg) {
        hist <- arg[["keras_history"]]
        if (is.null(hist)) {
          return(NA_character_)
        }
        lyr <- hist[[1L]]
        if (is.character(lyr)) {
          lyr
        } else {
          tryCatch(lyr$name, error = function(e) {
            as.character(lyr)[1L]
          })
        }
      },
      character(1L)
    )
    return(res[!is.na(res)])
  }

  # Old keras format: first_node is a list of [layer_name, node_idx, tensor_idx] triples
  if (
    is.list(first_node) &&
      length(first_node) > 0L &&
      is.list(first_node[[1L]])
  ) {
    return(vapply(
      first_node,
      function(spec) {
        n <- spec[[1L]]
        if (is.character(n)) n else as.character(n)[1L]
      },
      character(1L)
    ))
  }

  character(0L)
}


# Internal: topological/DAG traversal for Functional API models with merge layers.
orbital_keras_dag_impl <- function(
  x,
  mode,
  type,
  lvl,
  prefix,
  feature_names,
  all_weights,
  all_layers
) {
  # Build Dense weight map: layer_name → list(kernel, bias)
  # Use per-layer get_weights() so BatchNorm/PReLU weights don't skew indices.
  weight_map <- list()
  for (l in all_layers) {
    if (grepl("dense", tolower(class(l)[1L]))) {
      wts <- l$get_weights()
      if (length(wts) >= 2L) {
        weight_map[[l$name]] <- list(
          kernel = t(wts[[1L]]),
          bias = as.numeric(wts[[2L]])
        )
      }
    }
  }
  last_dense <- if (length(weight_map) > 0L) {
    tail(names(weight_map), 1L)
  } else {
    ""
  }

  n_in <- ncol(weight_map[[names(weight_map)[1L]]]$kernel)
  input_names <- if (!is.null(feature_names)) {
    feature_names
  } else {
    paste0("orbital_feature_", seq_len(n_in))
  }

  # Parse layer topology from model config
  cfg <- tryCatch(x$get_config(), error = function(e) NULL)
  layers_cfg <- if (!is.null(cfg)) {
    if (!is.null(cfg$config) && !is.null(cfg$config$layers)) {
      cfg$config$layers
    } else if (!is.null(cfg$layers)) {
      cfg$layers
    } else {
      list()
    }
  } else {
    list()
  }

  # Determine output layer names from model config (Functional API multi-output)
  raw_out <- if (!is.null(cfg)) {
    if (!is.null(cfg$config$output_layers)) {
      cfg$config$output_layers
    } else if (!is.null(cfg$output_layers)) {
      cfg$output_layers
    } else {
      NULL
    }
  } else {
    NULL
  }
  output_layer_names <- if (!is.null(raw_out) && length(raw_out) > 0L) {
    unique(vapply(
      raw_out,
      function(spec) {
        n <- if (is.list(spec)) spec[[1L]] else spec
        as.character(n)[1L]
      },
      character(1L)
    ))
  } else {
    last_dense # fallback: treat last Dense as the single output
  }

  topo_map <- vector("list", length(layers_cfg))
  for (k in seq_along(layers_cfg)) {
    lc <- layers_cfg[[k]]
    lname <- tryCatch(
      {
        n <- if (!is.null(lc$config) && !is.null(lc$config$name)) {
          lc$config$name
        } else {
          lc$name
        }
        as.character(n)[1L]
      },
      error = function(e) ""
    )
    if (nzchar(lname)) {
      topo_map[[lname]] <- .keras_parse_inbound(lc)
    }
  }

  # Expression register: layer_name → character vector of expression names
  expr_reg <- new.env(hash = TRUE, parent = emptyenv())
  all_exprs <- list()
  out_pre_act <- NULL # used for single-output models
  out_pre_act_map <- list() # layer_name -> pre_act, for multi-output models

  for (l in all_layers) {
    cls_orig <- class(l)[1L]
    cls <- tolower(cls_orig)
    lname <- l$name

    if (grepl("input", cls)) {
      assign(lname, input_names, envir = expr_reg)
    } else if (grepl("dense", cls)) {
      inbound <- topo_map[[lname]]
      if (is.null(inbound) || length(inbound) < 1L) {
        cli::cli_abort(
          "Dense layer {.val {lname}} has no inbound connections in model config."
        )
      }
      in_names <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      wt <- weight_map[[lname]]
      pre_act <- build_mlp_pre_act(wt$kernel, wt$bias, in_names)

      activation_config <- tryCatch(
        l$get_config()$activation,
        error = function(e) NULL
      )
      activation <- if (is.null(activation_config)) {
        "linear"
      } else if (is.list(activation_config)) {
        tolower(as.character(activation_config$class_name)[1L])
      } else {
        tolower(as.character(activation_config)[1L])
      }
      if (!nzchar(activation)) {
        activation <- "linear"
      }
      act_alpha <- if (
        is.list(activation_config) &&
          !is.null(activation_config[["config"]])
      ) {
        activation_config[["config"]][["alpha"]]
      } else {
        NULL
      }

      if (lname %in% output_layer_names) {
        out_pre_act_map[[lname]] <- pre_act
        if (lname == last_dense) out_pre_act <- pre_act
      } else {
        act_exprs <- vapply(
          pre_act,
          function(z) {
            activation_expr(activation, z, alpha = act_alpha)
          },
          character(1)
        )
        unit_names <- paste0(
          "orbital_mlp_",
          lname,
          "_h",
          seq_along(act_exprs)
        )
        all_exprs[[lname]] <- stats::setNames(act_exprs, unit_names)
        assign(lname, unit_names, envir = expr_reg)
      }
    } else if (grepl("\\badd\\b", cls, perl = TRUE)) {
      # Element-wise Add: supports skip / residual connections
      inbound <- topo_map[[lname]]
      if (is.null(inbound) || length(inbound) != 2L) {
        cli::cli_abort(
          "Keras Add layer {.val {lname}} must have exactly 2 inbound inputs, got {length(inbound)}."
        )
      }
      exprs_a <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      exprs_b <- get(inbound[2L], envir = expr_reg, inherits = FALSE)
      if (length(exprs_a) != length(exprs_b)) {
        cli::cli_abort(
          "Keras Add layer {.val {lname}}: both inputs must have the same width ({length(exprs_a)} vs {length(exprs_b)})."
        )
      }
      add_names <- paste0("orbital_", lname, "_h", seq_along(exprs_a))
      add_exprs <- paste0(
        "(",
        backtick(exprs_a),
        " + ",
        backtick(exprs_b),
        ")"
      )
      all_exprs[[lname]] <- stats::setNames(add_exprs, add_names)
      assign(lname, add_names, envir = expr_reg)
    } else if (grepl("concatenate", cls)) {
      # Concatenate: merge expression-name vectors from all inbound branches
      inbound <- topo_map[[lname]]
      if (is.null(inbound) || length(inbound) < 2L) {
        cli::cli_abort(
          "Keras Concatenate layer {.val {lname}} must have at least 2 inbound inputs."
        )
      }
      combined_names <- unlist(lapply(inbound, function(nm) {
        get(nm, envir = expr_reg, inherits = FALSE)
      }))
      assign(lname, combined_names, envir = expr_reg)
    } else if (grepl("batchnorm", cls)) {
      # BatchNormalization: ((x - mean) / sqrt(var + eps)) * gamma + beta
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      wts <- l$get_weights() # gamma, beta, moving_mean, moving_var
      gamma <- as.numeric(wts[[1L]])
      beta <- as.numeric(wts[[2L]])
      mn <- as.numeric(wts[[3L]])
      vr <- as.numeric(wts[[4L]])
      eps <- tryCatch(as.numeric(l$epsilon), error = function(e) 1e-5)
      unit_names <- paste0(
        "orbital_bn_",
        lname,
        "_h",
        seq_along(in_exprs)
      )
      bn_exprs <- vapply(
        seq_along(in_exprs),
        function(i) {
          xe <- backtick(in_exprs[[i]])
          glue::glue(
            "(({xe} - {format_numeric(mn[i])}) / sqrt({format_numeric(vr[i])} + {format_numeric(eps)})) * {format_numeric(gamma[i])} + {format_numeric(beta[i])}"
          )
        },
        character(1)
      )
      all_exprs[[lname]] <- stats::setNames(bn_exprs, unit_names)
      assign(lname, unit_names, envir = expr_reg)
    } else if (grepl("layernorm", cls)) {
      # LayerNormalization: per-row symbolic mean + var, then normalize
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      wts <- l$get_weights() # gamma, beta
      gamma <- as.numeric(wts[[1L]])
      beta <- as.numeric(wts[[2L]])
      n_feat <- length(in_exprs)
      eps <- tryCatch(as.numeric(l$epsilon), error = function(e) 1e-3)
      expr_bt <- backtick(in_exprs)
      mean_nm <- paste0("orbital_ln_mean_", lname)
      var_nm <- paste0("orbital_ln_var_", lname)
      mean_expr <- paste0(
        "(",
        paste(expr_bt, collapse = " + "),
        ") / ",
        n_feat
      )
      var_parts <- paste0("(", expr_bt, " - `", mean_nm, "`)^2")
      var_expr <- paste0(
        "(",
        paste(var_parts, collapse = " + "),
        ") / ",
        n_feat
      )
      unit_names <- paste0(
        "orbital_ln_",
        lname,
        "_h",
        seq_along(in_exprs)
      )
      norm_exprs <- vapply(
        seq_along(in_exprs),
        function(i) {
          xe <- backtick(in_exprs[[i]])
          glue::glue(
            "(({xe} - `{mean_nm}`) / sqrt(`{var_nm}` + {format_numeric(eps)})) * {format_numeric(gamma[i])} + {format_numeric(beta[i])}"
          )
        },
        character(1)
      )
      all_exprs[[lname]] <- c(
        stats::setNames(mean_expr, mean_nm),
        stats::setNames(var_expr, var_nm),
        stats::setNames(norm_exprs, unit_names)
      )
      assign(lname, unit_names, envir = expr_reg)
    } else if (grepl("prelu", cls)) {
      # PReLU: per-channel learnable negative slope
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      alpha_w <- as.numeric(unlist(l$get_weights()))
      unit_names <- paste0(
        "orbital_prelu_",
        lname,
        "_h",
        seq_along(in_exprs)
      )
      prelu_exprs <- vapply(
        seq_along(in_exprs),
        function(i) {
          xe <- backtick(in_exprs[[i]])
          a <- format_numeric(alpha_w[min(i, length(alpha_w))])
          glue::glue("dplyr::if_else({xe} >= 0, {xe}, {a} * {xe})")
        },
        character(1)
      )
      all_exprs[[lname]] <- stats::setNames(prelu_exprs, unit_names)
      assign(lname, unit_names, envir = expr_reg)
    } else if (grepl("dropout|flatten|reshape", cls)) {
      # Inference-transparent pass-through layers
      inbound <- topo_map[[lname]]
      if (
        !is.null(inbound) &&
          length(inbound) >= 1L &&
          exists(inbound[1L], envir = expr_reg, inherits = FALSE)
      ) {
        assign(
          lname,
          get(inbound[1L], envir = expr_reg, inherits = FALSE),
          envir = expr_reg
        )
      }
    } else if (grepl("leakyrelu", cls)) {
      # Standalone LeakyReLU layer (keras.src.layers.activation.leaky_relu.*)
      # Must be checked BEFORE the generic \bactivation\b branch because the
      # module path contains the word "activation".
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      slope <- tryCatch(
        as.numeric(l$get_config()$negative_slope),
        error = function(e) 0.01
      )
      if (is.null(slope) || is.na(slope)) {
        slope <- 0.01
      }
      unit_names <- paste0(
        "orbital_leakyrelu_",
        lname,
        "_h",
        seq_along(in_exprs)
      )
      lr_exprs <- vapply(
        in_exprs,
        function(e) activation_expr("leaky_relu", e, alpha = slope),
        character(1)
      )
      all_exprs[[lname]] <- stats::setNames(lr_exprs, unit_names)
      assign(lname, unit_names, envir = expr_reg)
    } else if (grepl("\\belu\\b", cls, perl = TRUE)) {
      # Standalone ELU layer (keras.src.layers.activation.elu.ELU).
      # The \belu\b word-boundary pattern does NOT match selu, celu, relu,
      # prelu, or leakyrelu, so this branch is safe to place after leakyrelu.
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      alpha_val <- tryCatch(
        as.numeric(l$get_config()$alpha),
        error = function(e) 1.0
      )
      if (is.null(alpha_val) || is.na(alpha_val)) {
        alpha_val <- 1.0
      }
      unit_names <- paste0(
        "orbital_elu_",
        lname,
        "_h",
        seq_along(in_exprs)
      )
      elu_exprs <- vapply(
        in_exprs,
        function(e) activation_expr("elu", e, alpha = alpha_val),
        character(1)
      )
      all_exprs[[lname]] <- stats::setNames(elu_exprs, unit_names)
      assign(lname, unit_names, envir = expr_reg)
    } else if (grepl("\\bactivation\\b", cls, perl = TRUE)) {
      # Standalone Activation layer: apply activation function to inbound expressions
      inbound <- topo_map[[lname]]
      if (is.null(inbound) || length(inbound) < 1L) {
        cli::cli_abort(
          "Activation layer {.val {lname}} has no inbound connections in model config."
        )
      }
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      act_cfg <- tryCatch(
        l$get_config()$activation,
        error = function(e) "linear"
      )
      activation <- if (is.null(act_cfg)) {
        "linear"
      } else if (is.list(act_cfg)) {
        tolower(as.character(act_cfg$class_name)[1L])
      } else {
        tolower(as.character(act_cfg)[1L])
      }
      if (!nzchar(activation)) {
        activation <- "linear"
      }
      if (activation %in% c("softmax", "log_softmax")) {
        expr_bt <- backtick(in_exprs)
        # Max-stabilisation: subtract max(logits) before exp() so that
        # exp() never overflows and results are numerically identical to
        # the unstabilised form (the shift cancels in both softmax and
        # log_softmax).
        max_nm <- paste0("orbital_act_sm_max_", lname)
        max_expr <- paste0(
          "do.call(pmax, list(",
          paste(expr_bt, collapse = ", "),
          "))"
        )
        sm_sum_nm <- paste0("orbital_act_sm_sum_", lname)
        sm_sum_expr <- paste0(
          "(",
          paste0(
            "exp(",
            expr_bt,
            " - `",
            max_nm,
            "`)",
            collapse = " + "
          ),
          ")"
        )
        unit_names <- paste0(
          "orbital_act_",
          lname,
          "_h",
          seq_along(in_exprs)
        )
        sm_exprs <- if (activation == "softmax") {
          vapply(
            seq_along(in_exprs),
            function(i) {
              paste0(
                "exp(",
                expr_bt[i],
                " - `",
                max_nm,
                "`) / `",
                sm_sum_nm,
                "`"
              )
            },
            character(1)
          )
        } else {
          # log_softmax = (x_i - max) - log(sum(exp(x_j - max)))
          vapply(
            seq_along(in_exprs),
            function(i) {
              paste0(
                "(",
                expr_bt[i],
                " - `",
                max_nm,
                "`) - log(`",
                sm_sum_nm,
                "`)"
              )
            },
            character(1)
          )
        }
        all_exprs[[lname]] <- c(
          stats::setNames(max_expr, max_nm),
          stats::setNames(sm_sum_expr, sm_sum_nm),
          stats::setNames(sm_exprs, unit_names)
        )
        assign(lname, unit_names, envir = expr_reg)
      } else {
        unit_names <- paste0(
          "orbital_act_",
          lname,
          "_h",
          seq_along(in_exprs)
        )
        act_exprs <- vapply(
          in_exprs,
          function(e) activation_expr(activation, e),
          character(1)
        )
        all_exprs[[lname]] <- stats::setNames(act_exprs, unit_names)
        assign(lname, unit_names, envir = expr_reg)
      }
    } else if (grepl("globalaveragepool", cls)) {
      # GlobalAveragePooling1D: reduce feature columns to their row-wise mean
      if (grepl("2d|3d", cls)) {
        cli::cli_abort(
          "GlobalAveragePooling2D/3D is not supported by orbital (requires spatial aggregation)."
        )
      }
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      n_f <- length(in_exprs)
      expr_bt <- backtick(in_exprs)
      gap_nm <- paste0("orbital_gap_", lname, "_1")
      gap_expr <- paste0(
        "(",
        paste(expr_bt, collapse = " + "),
        ") / ",
        n_f
      )
      all_exprs[[lname]] <- stats::setNames(gap_expr, gap_nm)
      assign(lname, gap_nm, envir = expr_reg)
    } else if (grepl("globalmaxpool", cls)) {
      # GlobalMaxPooling1D: reduce feature columns to their row-wise max
      if (grepl("2d|3d", cls)) {
        cli::cli_abort(
          "GlobalMaxPooling2D/3D is not supported by orbital (requires spatial aggregation)."
        )
      }
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      expr_bt <- backtick(in_exprs)
      gmp_nm <- paste0("orbital_gmp_", lname, "_1")
      gmp_expr <- paste0(
        "do.call(pmax, list(",
        paste(expr_bt, collapse = ", "),
        "))"
      )
      all_exprs[[lname]] <- stats::setNames(gmp_expr, gmp_nm)
      assign(lname, gmp_nm, envir = expr_reg)
    } else if (grepl("averagepooling1d", cls)) {
      # AveragePooling1D: row-wise mean of all feature columns
      if (grepl("2d|3d", cls)) {
        cli::cli_abort(
          "AveragePooling2D/3D is not supported by orbital (requires spatial aggregation)."
        )
      }
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      n_f <- length(in_exprs)
      expr_bt <- backtick(in_exprs)
      ap_nm <- paste0("orbital_avgpool1d_", lname, "_1")
      ap_expr <- paste0(
        "(",
        paste(expr_bt, collapse = " + "),
        ") / ",
        n_f
      )
      all_exprs[[lname]] <- stats::setNames(ap_expr, ap_nm)
      assign(lname, ap_nm, envir = expr_reg)
    } else if (grepl("maxpooling1d", cls)) {
      # MaxPooling1D: row-wise max of all feature columns
      if (grepl("2d|3d", cls)) {
        cli::cli_abort(
          "MaxPooling2D/3D is not supported by orbital (requires spatial aggregation)."
        )
      }
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      expr_bt <- backtick(in_exprs)
      mp_nm <- paste0("orbital_maxpool1d_", lname, "_1")
      mp_expr <- paste0(
        "do.call(pmax, list(",
        paste(expr_bt, collapse = ", "),
        "))"
      )
      all_exprs[[lname]] <- stats::setNames(mp_expr, mp_nm)
      assign(lname, mp_nm, envir = expr_reg)
    } else if (grepl("globalsumpooling", cls)) {
      # GlobalSumPooling1D: row-wise sum of all feature columns
      if (grepl("2d|3d", cls)) {
        cli::cli_abort(
          "GlobalSumPooling2D/3D is not supported by orbital (requires spatial aggregation)."
        )
      }
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      expr_bt <- backtick(in_exprs)
      gsp_nm <- paste0("orbital_gsp_", lname, "_1")
      gsp_expr <- paste0(
        "(",
        paste(expr_bt, collapse = " + "),
        ")"
      )
      all_exprs[[lname]] <- stats::setNames(gsp_expr, gsp_nm)
      assign(lname, gsp_nm, envir = expr_reg)
    } else if (grepl("instancenorm", cls)) {
      # InstanceNormalization: per-row normalize across features (same as LayerNorm for 1D)
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      wts <- l$get_weights() # gamma, beta
      gamma <- as.numeric(wts[[1L]])
      beta <- as.numeric(wts[[2L]])
      n_feat <- length(in_exprs)
      eps <- tryCatch(as.numeric(l$epsilon), error = function(e) 1e-3)
      expr_bt <- backtick(in_exprs)
      mean_nm <- paste0("orbital_in_mean_", lname)
      var_nm <- paste0("orbital_in_var_", lname)
      mean_expr <- paste0(
        "(",
        paste(expr_bt, collapse = " + "),
        ") / ",
        n_feat
      )
      var_parts <- paste0("(", expr_bt, " - `", mean_nm, "`)^2")
      var_expr <- paste0(
        "(",
        paste(var_parts, collapse = " + "),
        ") / ",
        n_feat
      )
      unit_names <- paste0(
        "orbital_in_",
        lname,
        "_h",
        seq_along(in_exprs)
      )
      norm_exprs <- vapply(
        seq_along(in_exprs),
        function(i) {
          xe <- backtick(in_exprs[[i]])
          glue::glue(
            "(({xe} - `{mean_nm}`) / sqrt(`{var_nm}` + {format_numeric(eps)})) * {format_numeric(gamma[i])} + {format_numeric(beta[i])}"
          )
        },
        character(1)
      )
      all_exprs[[lname]] <- c(
        stats::setNames(mean_expr, mean_nm),
        stats::setNames(var_expr, var_nm),
        stats::setNames(norm_exprs, unit_names)
      )
      assign(lname, unit_names, envir = expr_reg)
    } else if (grepl("groupnorm", cls)) {
      # GroupNormalization: normalize within each group of features
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      wts <- l$get_weights() # gamma, beta
      gamma <- as.numeric(wts[[1L]])
      beta <- as.numeric(wts[[2L]])
      n_feat <- length(in_exprs)
      eps <- tryCatch(as.numeric(l$epsilon), error = function(e) 1e-3)
      num_groups <- tryCatch(
        as.integer(l$get_config()$groups),
        error = function(e) 1L
      )
      if (is.null(num_groups) || is.na(num_groups)) {
        num_groups <- 1L
      }

      if (num_groups == 1L) {
        # Equivalent to LayerNorm: normalize all features together
        expr_bt <- backtick(in_exprs)
        mean_nm <- paste0("orbital_gn_mean_", lname)
        var_nm <- paste0("orbital_gn_var_", lname)
        mean_expr <- paste0(
          "(",
          paste(expr_bt, collapse = " + "),
          ") / ",
          n_feat
        )
        var_parts <- paste0("(", expr_bt, " - `", mean_nm, "`)^2")
        var_expr <- paste0(
          "(",
          paste(var_parts, collapse = " + "),
          ") / ",
          n_feat
        )
        unit_names <- paste0(
          "orbital_gn_",
          lname,
          "_h",
          seq_along(in_exprs)
        )
        norm_exprs <- vapply(
          seq_along(in_exprs),
          function(i) {
            xe <- backtick(in_exprs[[i]])
            glue::glue(
              "(({xe} - `{mean_nm}`) / sqrt(`{var_nm}` + {format_numeric(eps)})) * {format_numeric(gamma[i])} + {format_numeric(beta[i])}"
            )
          },
          character(1)
        )
        all_exprs[[lname]] <- c(
          stats::setNames(mean_expr, mean_nm),
          stats::setNames(var_expr, var_nm),
          stats::setNames(norm_exprs, unit_names)
        )
        assign(lname, unit_names, envir = expr_reg)
      } else if (num_groups == n_feat) {
        # Each group has 1 feature: per-feature normalization
        # With 1 element per group, mean = x_i and var = 0
        # output_i = (x_i - x_i) / sqrt(0 + eps) * gamma_i + beta_i = beta_i
        unit_names <- paste0(
          "orbital_gn_",
          lname,
          "_h",
          seq_along(in_exprs)
        )
        singleton_exprs <- vapply(
          seq_along(in_exprs),
          function(i) format_numeric(beta[i]),
          character(1)
        )
        all_exprs[[lname]] <- stats::setNames(
          singleton_exprs,
          unit_names
        )
        assign(lname, unit_names, envir = expr_reg)
      } else if (n_feat %% num_groups == 0L) {
        # General case: divide n_feat features into num_groups equal groups,
        # normalising within each group independently (matches Python impl).
        group_size <- n_feat %/% num_groups
        result_exprs <- character(0L)
        result_names <- character(0L)
        unit_nms_all <- character(0L)
        expr_bt <- backtick(in_exprs)

        for (g in seq_len(num_groups)) {
          g0 <- g - 1L
          start <- g0 * group_size + 1L
          end <- start + group_size - 1L
          g_cols <- expr_bt[start:end]
          k_g <- as.numeric(group_size)

          mean_nm <- paste0("orbital_gn_gmean_", lname, "_g", g)
          var_nm <- paste0("orbital_gn_gvar_", lname, "_g", g)

          mean_expr_g <- paste0(
            "(",
            paste(g_cols, collapse = " + "),
            ") / ",
            k_g
          )
          var_parts_g <- paste0("(", g_cols, " - `", mean_nm, "`)^2")
          var_expr_g <- paste0(
            "(",
            paste(var_parts_g, collapse = " + "),
            ") / ",
            k_g
          )

          unit_names_g <- paste0(
            "orbital_gn_",
            lname,
            "_h",
            seq(start, end)
          )
          unit_nms_all <- c(unit_nms_all, unit_names_g)

          # Capture loop variables for use inside vapply closure
          local({
            .mean_nm <- mean_nm
            .var_nm <- var_nm
            .start <- start
            .in_exprs <- in_exprs
            .gamma <- gamma
            .beta <- beta
            .eps <- eps
            .unit_names_g <- unit_names_g
            norm_exprs_g <<- vapply(
              seq_along(.unit_names_g),
              function(i_local) {
                c_idx <- .start + i_local - 1L
                xe <- backtick(.in_exprs[[c_idx]])
                glue::glue(
                  "(({xe} - `{.mean_nm}`) / sqrt(`{.var_nm}` + {format_numeric(.eps)})) * {format_numeric(.gamma[c_idx])} + {format_numeric(.beta[c_idx])}"
                )
              },
              character(1L)
            )
          })

          result_names <- c(
            result_names,
            mean_nm,
            var_nm,
            unit_names_g
          )
          result_exprs <- c(
            result_exprs,
            mean_expr_g,
            var_expr_g,
            norm_exprs_g
          )
        }

        all_exprs[[lname]] <- stats::setNames(
          result_exprs,
          result_names
        )
        assign(lname, unit_nms_all, envir = expr_reg)
      } else {
        cli::cli_abort(c(
          "Unsupported GroupNormalization configuration in Keras model: {.val {lname}} has {num_groups} groups for {n_feat} features.",
          "i" = "num_groups must evenly divide the number of features ({n_feat} %% {num_groups} != 0)."
        ))
      }
    } else if (grepl("\\bsoftmax\\b", cls, perl = TRUE)) {
      # Standalone Softmax layer: row-wise softmax normalisation (max-stabilised)
      # Subtracting the row-wise max before exp() prevents overflow for large logits
      # and does not change the result (the shift cancels in numerator and denominator).
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      expr_bt <- backtick(in_exprs)
      sm_max_nm <- paste0("orbital_sm_max_", lname)
      sm_max_expr <- paste0(
        "do.call(pmax, list(",
        paste(expr_bt, collapse = ", "),
        "))"
      )
      sm_sum_nm <- paste0("orbital_sm_sum_", lname)
      sm_sum_expr <- paste0(
        "(",
        paste0(
          "exp(",
          expr_bt,
          " - `",
          sm_max_nm,
          "`)",
          collapse = " + "
        ),
        ")"
      )
      unit_names <- paste0(
        "orbital_sm_",
        lname,
        "_h",
        seq_along(in_exprs)
      )
      sm_exprs <- vapply(
        seq_along(in_exprs),
        function(i) {
          paste0(
            "exp(",
            expr_bt[i],
            " - `",
            sm_max_nm,
            "`) / `",
            sm_sum_nm,
            "`"
          )
        },
        character(1)
      )
      all_exprs[[lname]] <- c(
        stats::setNames(sm_max_expr, sm_max_nm),
        stats::setNames(sm_sum_expr, sm_sum_nm),
        stats::setNames(sm_exprs, unit_names)
      )
      assign(lname, unit_names, envir = expr_reg)
    } else {
      cli::cli_abort(c(
        "Unsupported layer type in Keras Functional model: {.cls {cls_orig}}.",
        "i" = paste(
          "orbital supports: Dense, Add, Concatenate, BatchNormalization,",
          "LayerNormalization, InstanceNormalization, GroupNormalization,",
          "PReLU, GlobalAveragePooling1D, GlobalMaxPooling1D,",
          "AveragePooling1D, MaxPooling1D, GlobalSumPooling1D,",
          "Dropout, Flatten, Reshape, Activation, Softmax."
        ),
        "i" = "Please file an issue: {.url https://github.com/davidrsch/orbital/issues/14}"
      ))
    }
  }

  if (length(out_pre_act_map) == 0L) {
    cli::cli_abort(
      "Could not identify any output Dense layer in the Keras model."
    )
  }

  hidden_exprs <- unlist(all_exprs, use.names = TRUE)

  # Combine pre-activations from all output layers in config order
  all_out_pre_act <- unlist(
    out_pre_act_map[output_layer_names],
    use.names = FALSE
  )
  n_out <- length(all_out_pre_act)

  if (mode == "regression") {
    if (n_out == 1L) {
      c(hidden_exprs, stats::setNames(all_out_pre_act[1L], prefix))
    } else {
      # Multiple regression outputs: name them prefix_1, prefix_2, ...
      out_nms <- if (length(all_out_pre_act) == length(lvl) && !is.null(lvl)) {
        lvl
      } else {
        paste0(prefix, "_", seq_along(all_out_pre_act))
      }
      c(hidden_exprs, stats::setNames(all_out_pre_act, out_nms))
    }
  } else if (n_out == 1L) {
    c(
      hidden_exprs,
      binary_from_prob(
        activation_expr("sigmoid", all_out_pre_act[1L]),
        type,
        lvl
      )
    )
  } else {
    c(
      hidden_exprs,
      multiclass_from_logits(
        stats::setNames(all_out_pre_act, lvl),
        type,
        lvl
      )
    )
  }
}


orbital_keras_impl <- function(
  x,
  mode,
  type,
  lvl,
  prefix,
  feature_names = NULL
) {
  all_weights <- x$get_weights()
  all_layers <- x$layers

  # Detect layers that require DAG traversal (merge, normalisation, learned activations).
  non_linear_layers <- Filter(
    function(l) {
      cls <- tolower(class(l)[1L])
      grepl(
        "\\badd\\b|concatenate|batchnorm|layernorm|instancenorm|groupnorm|prelu|leakyrelu|\\belu\\b|globalaveragepool|globalmaxpool|averagepooling1d|maxpooling1d|globalsumpooling|\\bactivation\\b|\\bsoftmax\\b",
        cls,
        perl = TRUE
      ) &&
        !grepl(
          "dense|input|flatten|reshape|dropout",
          cls
        )
    },
    all_layers
  )

  if (length(non_linear_layers) > 0L) {
    return(
      orbital_keras_dag_impl(
        x,
        mode,
        type,
        lvl,
        prefix,
        feature_names,
        all_weights,
        all_layers
      )
    )
  }

  # Linear model: optimised sequential traversal (no topology introspection needed)
  n_dense <- length(all_weights) / 2L # each Dense layer has kernel + bias

  dense_layers <- Filter(
    function(l) grepl("dense", tolower(class(l)[1L])),
    all_layers
  )

  n_in <- ncol(all_weights[[1L]])
  input_names <- if (!is.null(feature_names)) {
    feature_names
  } else {
    paste0("orbital_feature_", seq_len(n_in))
  }

  all_exprs <- list()
  current_names <- input_names

  for (i in seq_len(n_dense)) {
    kernel <- t(all_weights[[2L * i - 1L]]) # (n_out_i x n_in_i)
    bias <- as.numeric(all_weights[[2L * i]])

    activation_config <- tryCatch(
      dense_layers[[i]]$get_config()$activation,
      error = function(e) NULL
    )
    activation <- if (is.null(activation_config)) {
      "linear"
    } else if (is.list(activation_config)) {
      tolower(as.character(activation_config$class_name)[1L])
    } else {
      tolower(as.character(activation_config)[1L])
    }
    if (!nzchar(activation)) {
      activation <- "linear"
    }
    act_alpha <- if (
      is.list(activation_config) &&
        !is.null(activation_config[["config"]])
    ) {
      activation_config[["config"]][["alpha"]]
    } else {
      NULL
    }

    pre_act <- build_mlp_pre_act(kernel, bias, current_names)

    if (i < n_dense) {
      act_exprs <- vapply(
        pre_act,
        function(z) activation_expr(activation, z, alpha = act_alpha),
        character(1)
      )
      layer_names <- paste0(
        "orbital_mlp_l",
        i,
        "_h",
        seq_len(nrow(kernel))
      )
      all_exprs[[i]] <- stats::setNames(act_exprs, layer_names)
      current_names <- layer_names
    } else {
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


#' @method orbital keras.engine.sequential.Sequential
#' @export
orbital.keras.engine.sequential.Sequential <- function(
  x,
  ...,
  mode = c("regression", "classification"),
  type = NULL,
  lvl = NULL,
  prefix = ".pred"
) {
  mode <- rlang::arg_match(mode)
  type <- default_type(type)
  extra <- list(...)
  orbital_keras_impl(
    x,
    mode = mode,
    type = type,
    lvl = lvl,
    prefix = prefix,
    feature_names = extra$feature_names
  )
}

#' @method orbital keras.src.models.sequential.Sequential
#' @export
orbital.keras.src.models.sequential.Sequential <- function(
  x,
  ...,
  mode = c("regression", "classification"),
  type = NULL,
  lvl = NULL,
  prefix = ".pred"
) {
  mode <- rlang::arg_match(mode)
  type <- default_type(type)
  extra <- list(...)
  orbital_keras_impl(
    x,
    mode = mode,
    type = type,
    lvl = lvl,
    prefix = prefix,
    feature_names = extra$feature_names
  )
}

#' @method orbital keras.src.models.functional.Functional
#' @export
orbital.keras.src.models.functional.Functional <- function(
  x,
  ...,
  mode = c("regression", "classification"),
  type = NULL,
  lvl = NULL,
  prefix = ".pred"
) {
  mode <- rlang::arg_match(mode)
  type <- default_type(type)
  extra <- list(...)
  orbital_keras_impl(
    x,
    mode = mode,
    type = type,
    lvl = lvl,
    prefix = prefix,
    feature_names = extra$feature_names
  )
}
