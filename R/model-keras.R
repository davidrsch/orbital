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

  # Helper: unroll a single LSTM or GRU layer into SQL expressions.
  # Returns list(nms, exprs, out_nms, all_H_nms, T, H).
  .unroll_rnn <- function(ul, ul_in_exprs, ul_pfx) {
    inner_cls <- tolower(class(ul)[1L])
    is_lstm_inner <- grepl("\\blstm\\b", inner_cls, perl = TRUE)
    ul_wts <- ul$get_weights()
    ul_cfg <- tryCatch(ul$get_config(), error = function(e) list())
    ul_return_seq <- isTRUE(
      tryCatch(as.logical(ul_cfg$return_sequences), error = function(e) FALSE)
    )
    ul_gate_act <- tryCatch(
      tolower(as.character(ul_cfg$recurrent_activation %||% "sigmoid")),
      error = function(e) "sigmoid"
    )
    ul_cell_act <- tryCatch(
      tolower(as.character(ul_cfg$activation %||% "tanh")),
      error = function(e) "tanh"
    )

    if (is_lstm_inner) {
      ul_kernel <- ul_wts[[1L]] # (I, 4H)
      ul_rkernel <- ul_wts[[2L]] # (H, 4H)
      ul_H <- as.integer(ncol(ul_kernel) / 4L)
      ul_I <- nrow(ul_kernel)
      ul_T <- as.integer(length(ul_in_exprs) / ul_I)
      ul_bias <- if (length(ul_wts) >= 3L) {
        as.numeric(ul_wts[[3L]])
      } else {
        numeric(4L * ul_H)
      }
      ul_g_offs <- c(0L, ul_H, 2L * ul_H, 3L * ul_H)
      ul_W <- lapply(seq_along(ul_g_offs), function(g) {
        t(ul_kernel[, (ul_g_offs[g] + 1L):(ul_g_offs[g] + ul_H)])
      })
      ul_R <- lapply(seq_along(ul_g_offs), function(g) {
        t(ul_rkernel[, (ul_g_offs[g] + 1L):(ul_g_offs[g] + ul_H)])
      })
      ul_B <- lapply(seq_along(ul_g_offs), function(g) {
        as.numeric(ul_bias[(ul_g_offs[g] + 1L):(ul_g_offs[g] + ul_H)])
      })

      nms <- character(0L)
      exprs <- character(0L)
      all_H_nms <- list()
      H_prev <- NULL
      C_prev <- NULL
      for (t in seq_len(ul_T)) {
        xt <- ul_in_exprs[((t - 1L) * ul_I + 1L):(t * ul_I)]
        pg <- lapply(seq_along(ul_g_offs), function(g) {
          inp <- build_mlp_pre_act(ul_W[[g]], ul_B[[g]], xt)
          if (is.null(H_prev)) {
            inp
          } else {
            paste0(
              "(",
              inp,
              " + ",
              build_mlp_pre_act(ul_R[[g]], numeric(ul_H), H_prev),
              ")"
            )
          }
        })
        aI <- vapply(
          pg[[1L]],
          function(e) activation_expr(ul_gate_act, e),
          character(1L)
        )
        aF <- vapply(
          pg[[2L]],
          function(e) activation_expr(ul_gate_act, e),
          character(1L)
        )
        aC <- vapply(
          pg[[3L]],
          function(e) activation_expr(ul_cell_act, e),
          character(1L)
        )
        aO <- vapply(
          pg[[4L]],
          function(e) activation_expr(ul_gate_act, e),
          character(1L)
        )
        C_nms <- paste0(ul_pfx, "_C_t", t, "_h", seq_len(ul_H))
        C_exprs_t <- if (is.null(C_prev)) {
          paste0("(", aI, " * ", aC, ")")
        } else {
          paste0("(", aF, " * ", backtick(C_prev), " + ", aI, " * ", aC, ")")
        }
        H_nms <- paste0(ul_pfx, "_H_t", t, "_h", seq_len(ul_H))
        H_exprs_t <- paste0("(", aO, " * tanh(", backtick(C_nms), "))")
        nms <- c(nms, C_nms, H_nms)
        exprs <- c(exprs, C_exprs_t, H_exprs_t)
        all_H_nms[[t]] <- H_nms
        H_prev <- H_nms
        C_prev <- C_nms
      }
      out_nms <- if (ul_return_seq) {
        unlist(all_H_nms, use.names = FALSE)
      } else {
        H_prev
      }
      list(
        nms = nms,
        exprs = exprs,
        out_nms = out_nms,
        all_H_nms = all_H_nms,
        T = ul_T,
        H = ul_H
      )
    } else {
      # GRU
      ul_kernel <- ul_wts[[1L]] # (I, 3H)
      ul_rkernel <- ul_wts[[2L]] # (H, 3H)
      ul_H <- as.integer(ncol(ul_kernel) / 3L)
      ul_I <- nrow(ul_kernel)
      ul_T <- as.integer(length(ul_in_exprs) / ul_I)
      if (length(ul_wts) >= 3L) {
        raw_b <- ul_wts[[3L]]
        if (!is.null(dim(raw_b)) && length(dim(raw_b)) == 2L) {
          ul_b_inp <- as.numeric(raw_b[1L, ])
          ul_b_rec <- as.numeric(raw_b[2L, ])
        } else {
          ba <- as.numeric(raw_b)
          ul_b_inp <- ba[seq_len(3L * ul_H)]
          ul_b_rec <- ba[seq_len(3L * ul_H) + 3L * ul_H]
        }
      } else {
        ul_b_inp <- numeric(3L * ul_H)
        ul_b_rec <- numeric(3L * ul_H)
      }
      ul_reset_after <- isTRUE(
        tryCatch(as.logical(ul_cfg$reset_after), error = function(e) TRUE)
      )
      ul_g_offs <- c(0L, ul_H, 2L * ul_H)
      ul_B_comb <- lapply(seq_along(ul_g_offs), function(g) {
        idx <- (ul_g_offs[g] + 1L):(ul_g_offs[g] + ul_H)
        ul_b_inp[idx] + ul_b_rec[idx]
      })
      ul_W <- lapply(seq_along(ul_g_offs), function(g) {
        t(ul_kernel[, (ul_g_offs[g] + 1L):(ul_g_offs[g] + ul_H)])
      })
      ul_R <- lapply(seq_along(ul_g_offs), function(g) {
        t(ul_rkernel[, (ul_g_offs[g] + 1L):(ul_g_offs[g] + ul_H)])
      })

      nms <- character(0L)
      exprs <- character(0L)
      all_H_nms <- list()
      H_prev <- NULL
      for (t in seq_len(ul_T)) {
        xt <- ul_in_exprs[((t - 1L) * ul_I + 1L):(t * ul_I)]
        z_pre <- {
          inp <- build_mlp_pre_act(ul_W[[1L]], ul_B_comb[[1L]], xt)
          if (is.null(H_prev)) {
            inp
          } else {
            paste0(
              "(",
              inp,
              " + ",
              build_mlp_pre_act(ul_R[[1L]], numeric(ul_H), H_prev),
              ")"
            )
          }
        }
        r_pre <- {
          inp <- build_mlp_pre_act(ul_W[[2L]], ul_B_comb[[2L]], xt)
          if (is.null(H_prev)) {
            inp
          } else {
            paste0(
              "(",
              inp,
              " + ",
              build_mlp_pre_act(ul_R[[2L]], numeric(ul_H), H_prev),
              ")"
            )
          }
        }
        z_nms_t <- paste0(ul_pfx, "_z_t", t, "_h", seq_len(ul_H))
        r_nms_t <- paste0(ul_pfx, "_r_t", t, "_h", seq_len(ul_H))
        z_exprs_t <- vapply(
          z_pre,
          function(e) activation_expr(ul_gate_act, e),
          character(1L)
        )
        r_exprs_t <- vapply(
          r_pre,
          function(e) activation_expr(ul_gate_act, e),
          character(1L)
        )
        h_idx <- (ul_g_offs[3L] + 1L):(ul_g_offs[3L] + ul_H)
        h_inp <- build_mlp_pre_act(
          ul_W[[3L]],
          if (ul_reset_after) ul_b_inp[h_idx] else ul_B_comb[[3L]],
          xt
        )
        ul_R_h <- ul_R[[3L]] # (H, H)
        ht_exprs_t <- vapply(
          seq_len(ul_H),
          function(h) {
            if (is.null(H_prev)) {
              activation_expr(ul_cell_act, h_inp[h])
            } else if (ul_reset_after) {
              rec_sum <- paste(
                paste0(backtick(H_prev), " * ", format_numeric(ul_R_h[h, ])),
                collapse = " + "
              )
              rec_wb <- paste0(
                "(",
                rec_sum,
                " + ",
                format_numeric(ul_b_rec[h_idx[h]]),
                ")"
              )
              activation_expr(
                ul_cell_act,
                paste0(
                  "(",
                  h_inp[h],
                  " + ",
                  backtick(r_nms_t[h]),
                  " * ",
                  rec_wb,
                  ")"
                )
              )
            } else {
              coupled <- paste(
                paste0(
                  backtick(r_nms_t),
                  " * ",
                  backtick(H_prev),
                  " * ",
                  format_numeric(ul_R_h[h, ])
                ),
                collapse = " + "
              )
              activation_expr(
                ul_cell_act,
                paste0("(", h_inp[h], " + ", coupled, ")")
              )
            }
          },
          character(1L)
        )
        ht_nms_t <- paste0(ul_pfx, "_ht_t", t, "_h", seq_len(ul_H))
        H_nms <- paste0(ul_pfx, "_H_t", t, "_h", seq_len(ul_H))
        H_cur_exprs <- if (is.null(H_prev)) {
          paste0("((1 - ", backtick(z_nms_t), ") * ", backtick(ht_nms_t), ")")
        } else {
          paste0(
            "((1 - ",
            backtick(z_nms_t),
            ") * ",
            backtick(ht_nms_t),
            " + ",
            backtick(z_nms_t),
            " * ",
            backtick(H_prev),
            ")"
          )
        }
        nms <- c(nms, z_nms_t, r_nms_t, ht_nms_t, H_nms)
        exprs <- c(exprs, z_exprs_t, r_exprs_t, ht_exprs_t, H_cur_exprs)
        all_H_nms[[t]] <- H_nms
        H_prev <- H_nms
      }
      out_nms <- if (ul_return_seq) {
        unlist(all_H_nms, use.names = FALSE)
      } else {
        H_prev
      }
      list(
        nms = nms,
        exprs = exprs,
        out_nms = out_nms,
        all_H_nms = all_H_nms,
        T = ul_T,
        H = ul_H
      )
    }
  }

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
        if (!activation %in% c("linear", "softmax", "log_softmax", "sigmoid")) {
          cli::cli_warn(c(
            "Dense output layer {.val {lname}} has activation = {.val {activation}} which orbital will override.",
            "i" = "orbital applies its own output transform based on {.arg mode}; the Keras activation on the output layer is ignored."
          ))
        }
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
      # Element-wise Add: supports skip / residual connections (>=2 inputs)
      inbound <- topo_map[[lname]]
      if (is.null(inbound) || length(inbound) < 2L) {
        cli::cli_abort(
          "Keras Add layer {.val {lname}} must have at least 2 inbound inputs, got {length(inbound)}."
        )
      }
      all_inbound_exprs <- lapply(inbound, function(nm) {
        get(nm, envir = expr_reg, inherits = FALSE)
      })
      widths <- lengths(all_inbound_exprs)
      if (length(unique(widths)) != 1L) {
        cli::cli_abort(
          "Keras Add layer {.val {lname}}: all inputs must have the same width (got: {paste(widths, collapse = ', ')})."
        )
      }
      add_names <- paste0("orbital_", lname, "_h", seq_len(widths[1L]))
      add_exprs <- vapply(
        seq_len(widths[1L]),
        function(i) {
          terms <- vapply(
            all_inbound_exprs,
            function(e) backtick(e[i]),
            character(1L)
          )
          paste0("(", paste(terms, collapse = " + "), ")")
        },
        character(1L)
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
      eps <- tryCatch(as.numeric(l$epsilon), error = function(e) 1e-3)
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
    } else if (grepl("\\brelu\\b", cls, perl = TRUE)) {
      # Standalone ReLU layer (keras.layers.ReLU()): must be checked BEFORE the
      # generic \bactivation\b branch because the module path contains "activation".
      # Keras ReLU config: negative_slope (default 0), max_value (default NULL),
      # threshold (default 0).  Full formula:
      #   base = if_else(x >= threshold, x - threshold, neg_slope * (x - threshold))
      #   out  = if max_value set: if_else(base > max_value, max_value, base)
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      ns <- tryCatch(
        as.numeric(l$get_config()$negative_slope),
        error = function(e) 0.0
      )
      thr <- tryCatch(
        as.numeric(l$get_config()$threshold),
        error = function(e) 0.0
      )
      if (is.null(ns) || length(ns) == 0L || is.na(ns)) {
        ns <- 0.0
      }
      if (is.null(thr) || length(thr) == 0L || is.na(thr)) {
        thr <- 0.0
      }
      mv_raw <- tryCatch(l$get_config()$max_value, error = function(e) NULL)
      mv_finite <- !is.null(mv_raw) &&
        length(mv_raw) > 0L &&
        !is.na(suppressWarnings(as.numeric(mv_raw)[[1L]]))
      mv <- if (mv_finite) as.numeric(mv_raw)[[1L]] else NA_real_
      unit_names <- paste0("orbital_relu_", lname, "_h", seq_along(in_exprs))
      relu_exprs <- vapply(
        in_exprs,
        function(e) {
          xe <- backtick(e)
          thr_f <- format_numeric(thr)
          ns_f <- format_numeric(ns)
          base <- if (thr == 0.0 && ns == 0.0) {
            glue::glue("dplyr::if_else({xe} >= 0, {xe}, 0)")
          } else if (thr == 0.0) {
            glue::glue("dplyr::if_else({xe} >= 0, {xe}, {ns_f} * {xe})")
          } else {
            glue::glue(
              "dplyr::if_else({xe} >= {thr_f}, {xe} - {thr_f}, {ns_f} * ({xe} - {thr_f}))"
            )
          }
          if (mv_finite) {
            mv_f <- format_numeric(mv)
            base <- glue::glue(
              "dplyr::if_else({base} > {mv_f}, {mv_f}, {base})"
            )
          }
          base
        },
        character(1L)
      )
      all_exprs[[lname]] <- stats::setNames(relu_exprs, unit_names)
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
      # AveragePooling1D: row-wise mean of all feature columns.
      # Global pooling (pool_size == T_in) is supported; windowed is not.
      if (grepl("2d|3d", cls)) {
        cli::cli_abort(
          "AveragePooling2D/3D is not supported by orbital (requires spatial aggregation)."
        )
      }
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      n_f <- length(in_exprs)
      pool_size <- tryCatch(
        as.integer(l$get_config()$pool_size),
        error = function(e) NA_integer_
      )
      if (!is.na(pool_size) && pool_size < n_f) {
        cli::cli_abort(c(
          "Windowed AveragePooling1D is not yet supported by orbital.",
          "i" = paste0(
            "pool_size = ",
            pool_size,
            ", input columns = ",
            n_f,
            ". ",
            "Only global pooling (pool_size covering all time steps) is supported."
          )
        ))
      }
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
      # MaxPooling1D: row-wise max of all feature columns.
      # Global pooling (pool_size == T_in) is supported; windowed is not.
      if (grepl("2d|3d", cls)) {
        cli::cli_abort(
          "MaxPooling2D/3D is not supported by orbital (requires spatial aggregation)."
        )
      }
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      pool_size <- tryCatch(
        as.integer(l$get_config()$pool_size),
        error = function(e) NA_integer_
      )
      if (!is.na(pool_size) && pool_size < length(in_exprs)) {
        cli::cli_abort(c(
          "Windowed MaxPooling1D is not yet supported by orbital.",
          "i" = paste0(
            "pool_size = ",
            pool_size,
            ", input columns = ",
            length(in_exprs),
            ". ",
            "Only global pooling (pool_size covering all time steps) is supported."
          )
        ))
      }
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
      # GlobalSumPooling1D: row-wise sum of all feature columns.
      # NOTE: GlobalSumPooling1D is not a standard Keras 3 layer; it exists
      # only in keras_cv (keras-cv package). This branch is effectively dead
      # code for stock Keras 3 models. If you are using keras_cv, verify that
      # the class name it produces contains "globalsumpooling"; otherwise this
      # branch will never be reached.
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
      # InstanceNormalization: per-row normalize across features (same as LayerNorm for 1D).
      # NOTE: InstanceNormalization is not a standard Keras 3 layer; it is
      # provided by keras_cv. The class path may differ across keras_cv
      # versions — verify the detected class string contains "instancenorm"
      # before relying on this branch.
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
    } else if (grepl("rmsnormalization", cls)) {
      # RMSNormalization: per-row normalize using root mean square (no mean subtraction).
      # rms  = sqrt((1/C) * sum_c(x_c^2) + epsilon)
      # y_c  = (x_c / rms) * scale_c
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      wts <- l$get_weights() # scale only (no bias)
      gamma <- as.numeric(wts[[1L]])
      n_feat <- length(in_exprs)
      eps <- tryCatch(as.numeric(l$epsilon), error = function(e) 1e-6)
      expr_bt <- backtick(in_exprs)
      rms_nm <- paste0("orbital_rms_", lname)
      rms_sq_parts <- paste0(expr_bt, "^2")
      rms_expr <- paste0(
        "sqrt((",
        paste(rms_sq_parts, collapse = " + "),
        ") / ",
        n_feat,
        " + ",
        format_numeric(eps),
        ")"
      )
      unit_names <- paste0(
        "orbital_rms_",
        lname,
        "_h",
        seq_along(in_exprs)
      )
      norm_exprs <- vapply(
        seq_along(in_exprs),
        function(i) {
          xe <- backtick(in_exprs[[i]])
          glue::glue(
            "({xe} / `{rms_nm}`) * {format_numeric(gamma[i])}"
          )
        },
        character(1)
      )
      all_exprs[[lname]] <- c(
        stats::setNames(rms_expr, rms_nm),
        stats::setNames(norm_exprs, unit_names)
      )
      assign(lname, unit_names, envir = expr_reg)
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
    } else if (
      grepl("conv1d", cls) && !grepl("depthwise|separable|2d|3d", cls)
    ) {
      # Conv1D ─ sliding-window 1-D convolution.
      # Keras weight layout:
      #   kernel  : (kernel_size, in_channels, filters)
      #   bias    : (filters,)  [optional]
      # orbital input convention : C_in × T_in flat columns, channel-major.
      #   in_exprs[(c-1)*T_in + t] = orbital feature for channel c, timestep t (1-indexed).
      # orbital output convention: T_out × C_out flat columns, time-step major.
      #   out_names[(p-1)*C_out + f] = filter f at output timestep p (1-indexed).
      #   This matches Keras 3's (batch, T_out, C_out) row-major flattening.
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

      wts <- l$get_weights()
      kern <- wts[[1L]] # (kW, C_in, C_out)
      k_w <- dim(kern)[1L]
      c_in <- dim(kern)[2L]
      c_out <- dim(kern)[3L]
      bias_v <- if (length(wts) >= 2L) as.numeric(wts[[2L]]) else numeric(c_out)

      cfg_l <- tryCatch(l$get_config(), error = function(e) list())
      stride <- tryCatch(as.integer(cfg_l$strides[[1L]]), error = function(e) {
        1L
      })
      dilation <- tryCatch(
        as.integer(cfg_l$dilation_rate[[1L]]),
        error = function(e) 1L
      )
      pad_type <- tryCatch(
        tolower(as.character(cfg_l$padding)),
        error = function(e) "valid"
      )
      if (is.null(stride) || is.na(stride)) {
        stride <- 1L
      }
      if (is.null(dilation) || is.na(dilation)) {
        dilation <- 1L
      }
      if (!nzchar(pad_type %||% "")) {
        pad_type <- "valid"
      }

      n_total <- length(in_exprs)
      w_in <- as.integer(n_total / c_in)
      k_eff <- dilation * (k_w - 1L) + 1L

      if (pad_type == "same") {
        w_out <- as.integer(ceiling(w_in / stride))
        pad_total <- max(0L, (w_out - 1L) * stride + k_eff - w_in)
        pad_l <- as.integer(floor(pad_total / 2L))
      } else {
        w_out <- as.integer(floor((w_in - k_eff) / stride) + 1L)
        pad_l <- 0L
      }

      conv_nms <- character(0L)
      conv_exprs <- character(0L)
      for (p in seq_len(w_out)) {
        p0 <- p - 1L
        for (f in seq_len(c_out)) {
          terms <- character(0L)
          for (c in seq_len(c_in)) {
            for (k in seq_len(k_w)) {
              k0 <- k - 1L
              w_pos <- p0 * stride + k0 * dilation - pad_l
              if (w_pos >= 0L && w_pos < w_in) {
                feat_nm <- in_exprs[(c - 1L) * w_in + w_pos + 1L]
                wt_val <- format_numeric(kern[k, c, f])
                terms <- c(
                  terms,
                  paste0("(", backtick(feat_nm), " * ", wt_val, ")")
                )
              }
            }
          }
          b_str <- format_numeric(bias_v[f])
          expr_str <- if (length(terms) == 0L) {
            b_str
          } else {
            paste0("(", paste(c(terms, b_str), collapse = " + "), ")")
          }
          nm <- paste0("orbital_conv_", lname, "_p", p, "_f", f)
          conv_nms <- c(conv_nms, nm)
          conv_exprs <- c(conv_exprs, expr_str)
        }
      }
      all_exprs[[lname]] <- stats::setNames(conv_exprs, conv_nms)
      assign(lname, conv_nms, envir = expr_reg)
    } else if (grepl("\\blstm\\b", cls, perl = TRUE)) {
      # LSTM ─ unrolled for fixed-length sequences.
      # Keras weight layout (IFCO gate order, columns 1:H = I, H+1:2H = F, ...):
      #   kernel           : (input_size,  4 * units)
      #   recurrent_kernel : (units,        4 * units)
      #   bias             : (4 * units,)  [optional, Keras sums input+recurrent biases]
      # orbital input: T * I flat columns (time-step major).
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      wts <- l$get_weights() # kernel, recurrent_kernel [, bias]

      kernel <- wts[[1L]] # (I, 4H)
      rkernel <- wts[[2L]] # (H, 4H)
      H <- as.integer(ncol(kernel) / 4L)
      I_feat <- nrow(kernel)
      T_len <- as.integer(length(in_exprs) / I_feat)
      bias_v <- if (length(wts) >= 3L) {
        as.numeric(wts[[3L]])
      } else {
        numeric(4L * H)
      }

      # Get activation config (defaults: sigmoid for gates I/F/O, tanh for C)
      cfg_l <- tryCatch(l$get_config(), error = function(e) list())
      if (
        isTRUE(tryCatch(as.logical(cfg_l$stateful), error = function(e) FALSE))
      ) {
        cli::cli_abort(c(
          "LSTM layer {.val {lname}}: stateful = TRUE is not supported by orbital.",
          "i" = "Only stateless LSTMs (stateful = FALSE, the Keras default) can be unrolled into SQL."
        ))
      }
      gate_act <- tryCatch(
        tolower(as.character(cfg_l$recurrent_activation %||% "sigmoid")),
        error = function(e) "sigmoid"
      )
      cell_act <- tryCatch(
        tolower(as.character(cfg_l$activation %||% "tanh")),
        error = function(e) "tanh"
      )
      return_seq <- isTRUE(
        tryCatch(as.logical(cfg_l$return_sequences), error = function(e) FALSE)
      )

      # IFCO gate offsets (0-indexed column starts):
      # I=0, F=H, C=2H, O=3H
      g_offs <- c(0L, H, 2L * H, 3L * H)

      # Transposed weight matrices for gate g (0-indexed g=0..3):
      #   W_gates[[g+1]] : (H × I_feat), used with build_mlp_pre_act
      #   R_gates[[g+1]] : (H × H),      used with build_mlp_pre_act
      W_gates <- lapply(seq_along(g_offs), function(g) {
        t(kernel[, (g_offs[g] + 1L):(g_offs[g] + H)])
      })
      R_gates <- lapply(seq_along(g_offs), function(g) {
        t(rkernel[, (g_offs[g] + 1L):(g_offs[g] + H)])
      })
      B_gates <- lapply(seq_along(g_offs), function(g) {
        as.numeric(bias_v[(g_offs[g] + 1L):(g_offs[g] + H)])
      })

      all_lstm_nms <- character(0L)
      all_lstm_exprs <- character(0L)
      all_H_nms <- list() # collect per-timestep H names for return_sequences

      H_prev_nms <- NULL # NULL = zero initial hidden state
      C_prev_nms <- NULL # NULL = zero initial cell state

      for (t in seq_len(T_len)) {
        x_t_exprs <- in_exprs[((t - 1L) * I_feat + 1L):(t * I_feat)]

        # Pre-activations for all four gates (IFCO order)
        pre_gates <- lapply(seq_along(g_offs), function(g) {
          inp <- build_mlp_pre_act(W_gates[[g]], B_gates[[g]], x_t_exprs)
          if (is.null(H_prev_nms)) {
            inp # H_prev = 0 → recurrent contribution is 0
          } else {
            rec <- build_mlp_pre_act(R_gates[[g]], numeric(H), H_prev_nms)
            paste0("(", inp, " + ", rec, ")")
          }
        })
        # pre_gates[[1]] = I gate, [[2]] = F gate, [[3]] = C gate, [[4]] = O gate

        act_I <- vapply(
          pre_gates[[1L]],
          function(e) activation_expr(gate_act, e),
          character(1L)
        )
        act_F <- vapply(
          pre_gates[[2L]],
          function(e) activation_expr(gate_act, e),
          character(1L)
        )
        act_C <- vapply(
          pre_gates[[3L]],
          function(e) activation_expr(cell_act, e),
          character(1L)
        )
        act_O <- vapply(
          pre_gates[[4L]],
          function(e) activation_expr(gate_act, e),
          character(1L)
        )

        # Cell state C_t = F_t * C_{t-1} + I_t * C̃_t
        C_cur_nms <- paste0("orbital_lstm_", lname, "_C_t", t, "_h", seq_len(H))
        C_cur_exprs <- if (is.null(C_prev_nms)) {
          paste0("(", act_I, " * ", act_C, ")")
        } else {
          paste0(
            "(",
            act_F,
            " * ",
            backtick(C_prev_nms),
            " + ",
            act_I,
            " * ",
            act_C,
            ")"
          )
        }

        # Hidden state H_t = O_t * tanh(C_t)
        H_cur_nms <- paste0("orbital_lstm_", lname, "_H_t", t, "_h", seq_len(H))
        H_cur_exprs <- paste0("(", act_O, " * tanh(", backtick(C_cur_nms), "))")

        all_lstm_nms <- c(all_lstm_nms, C_cur_nms, H_cur_nms)
        all_lstm_exprs <- c(all_lstm_exprs, C_cur_exprs, H_cur_exprs)
        all_H_nms[[t]] <- H_cur_nms

        H_prev_nms <- H_cur_nms
        C_prev_nms <- C_cur_nms
      }

      all_exprs[[lname]] <- stats::setNames(all_lstm_exprs, all_lstm_nms)
      out_nms <- if (return_seq) {
        unlist(all_H_nms, use.names = FALSE)
      } else {
        H_prev_nms # last timestep's hidden state
      }
      assign(lname, out_nms, envir = expr_reg)
    } else if (grepl("\\bgru\\b", cls, perl = TRUE)) {
      # GRU ─ unrolled for fixed-length sequences.
      # Keras weight layout (ZRH gate order):
      #   kernel           : (input_size, 3 * units)
      #   recurrent_kernel : (units,       3 * units)
      #   bias             : (2, 3 * units)  row 1 = input bias, row 2 = recurrent bias
      # Gate equations (ONNX default, linear_before_reset = 0):
      #   z_t = f(x@Wz + H_prev@Rz + bz)
      #   r_t = f(x@Wr + H_prev@Rr + br)
      #   h̃_t = g(x@Wh + (r_t ⊙ H_prev)@Rh + bh)
      #   H_t = (1 − z_t) ⊙ h̃_t + z_t ⊙ H_prev
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      wts <- l$get_weights() # kernel, recurrent_kernel [, bias]

      kernel <- wts[[1L]] # (I, 3H)
      rkernel <- wts[[2L]] # (H, 3H)
      H <- as.integer(ncol(kernel) / 3L)
      I_feat <- nrow(kernel)
      T_len <- as.integer(length(in_exprs) / I_feat)

      # Bias: (2, 3H) matrix or flat (6H) vector
      if (length(wts) >= 3L) {
        raw_b <- wts[[3L]]
        if (!is.null(dim(raw_b)) && length(dim(raw_b)) == 2L) {
          b_input <- as.numeric(raw_b[1L, ])
          b_recur <- as.numeric(raw_b[2L, ])
        } else {
          ba <- as.numeric(raw_b)
          b_input <- ba[seq_len(3L * H)]
          b_recur <- ba[seq_len(3L * H) + 3L * H]
        }
      } else {
        b_input <- numeric(3L * H)
        b_recur <- numeric(3L * H)
      }

      # Get activation config (defaults: sigmoid for Z/R, tanh for H-tilde)
      cfg_l <- tryCatch(l$get_config(), error = function(e) list())
      if (
        isTRUE(tryCatch(as.logical(cfg_l$stateful), error = function(e) FALSE))
      ) {
        cli::cli_abort(c(
          "GRU layer {.val {lname}}: stateful = TRUE is not supported by orbital.",
          "i" = "Only stateless GRUs (stateful = FALSE, the Keras default) can be unrolled into SQL."
        ))
      }
      gate_act <- tryCatch(
        tolower(as.character(cfg_l$recurrent_activation %||% "sigmoid")),
        error = function(e) "sigmoid"
      )
      cell_act <- tryCatch(
        tolower(as.character(cfg_l$activation %||% "tanh")),
        error = function(e) "tanh"
      )
      return_seq <- isTRUE(
        tryCatch(as.logical(cfg_l$return_sequences), error = function(e) FALSE)
      )
      # When reset_after = TRUE (the Keras 3 default), the recurrent bias for
      # the h-tilde gate is applied INSIDE the reset-gate multiplication:
      #   h̃_t = g(x@Wh + b_hx + r_t[h] * (H_prev@Rh[h] + b_hh[h]))
      # When reset_after = FALSE (legacy), the combined bias goes outside:
      #   h̃_t = g((r ⊙ H_prev)@Rh + x@Wh + b_h)
      reset_after <- isTRUE(
        tryCatch(as.logical(cfg_l$reset_after), error = function(e) TRUE)
      )

      # ZRH gate offsets (0-indexed column starts): Z=0, R=H, H-tilde=2H
      g_offs <- c(0L, H, 2L * H)

      # Combined bias (input + recurrent) per gate
      B_gates <- lapply(seq_along(g_offs), function(g) {
        idx <- (g_offs[g] + 1L):(g_offs[g] + H)
        b_input[idx] + b_recur[idx]
      })

      # Transposed weight matrices: W_gates[[g]][h, i] = weight from input i to unit h
      W_gates <- lapply(seq_along(g_offs), function(g) {
        t(kernel[, (g_offs[g] + 1L):(g_offs[g] + H)])
      })
      # R_gates[[g]][h, hid] = weight from recurrent unit hid to output unit h
      R_gates <- lapply(seq_along(g_offs), function(g) {
        t(rkernel[, (g_offs[g] + 1L):(g_offs[g] + H)])
      })

      all_gru_nms <- character(0L)
      all_gru_exprs <- character(0L)
      all_H_nms <- list()

      H_prev_nms <- NULL # NULL = zero initial hidden state

      for (t in seq_len(T_len)) {
        x_t_exprs <- in_exprs[((t - 1L) * I_feat + 1L):(t * I_feat)]

        # z gate (update gate)
        z_pre <- {
          inp <- build_mlp_pre_act(W_gates[[1L]], B_gates[[1L]], x_t_exprs)
          if (is.null(H_prev_nms)) {
            inp
          } else {
            rec <- build_mlp_pre_act(R_gates[[1L]], numeric(H), H_prev_nms)
            paste0("(", inp, " + ", rec, ")")
          }
        }
        z_nms <- paste0("orbital_gru_", lname, "_z_t", t, "_h", seq_len(H))
        z_exprs <- vapply(
          z_pre,
          function(e) activation_expr(gate_act, e),
          character(1L)
        )

        # r gate (reset gate)
        r_pre <- {
          inp <- build_mlp_pre_act(W_gates[[2L]], B_gates[[2L]], x_t_exprs)
          if (is.null(H_prev_nms)) {
            inp
          } else {
            rec <- build_mlp_pre_act(R_gates[[2L]], numeric(H), H_prev_nms)
            paste0("(", inp, " + ", rec, ")")
          }
        }
        r_nms <- paste0("orbital_gru_", lname, "_r_t", t, "_h", seq_len(H))
        r_exprs <- vapply(
          r_pre,
          function(e) activation_expr(gate_act, e),
          character(1L)
        )

        # h̃ gate
        # reset_after=TRUE  (Keras 3 default): h̃ = g(x@Wh + b_hx + r[h]*(H_prev@Rh + b_hh))
        # reset_after=FALSE (legacy):          h̃ = g((r ⊙ H_prev)@Rh + x@Wh + b_h)
        R_h <- R_gates[[3L]] # (H, H): R_h[h, hid] = recurrent weight
        h_idx <- (g_offs[3L] + 1L):(g_offs[3L] + H)
        h_inp <- build_mlp_pre_act(
          W_gates[[3L]],
          if (reset_after) b_input[h_idx] else B_gates[[3L]],
          x_t_exprs
        )

        h_tilde_exprs <- vapply(
          seq_len(H),
          function(h) {
            if (is.null(H_prev_nms)) {
              # H_prev = 0 → recurrent term vanishes entirely
              activation_expr(cell_act, h_inp[h])
            } else if (reset_after) {
              # Correct reset_after=TRUE formula:
              # r[h] * (\sum_j H_prev[j]*R_h[h,j] + b_hh[h])
              rec_sum <- paste(
                paste0(backtick(H_prev_nms), " * ", format_numeric(R_h[h, ])),
                collapse = " + "
              )
              rec_with_bias <- paste0(
                "(",
                rec_sum,
                " + ",
                format_numeric(b_recur[h_idx[h]]),
                ")"
              )
              coupled <- paste0(backtick(r_nms[h]), " * ", rec_with_bias)
              activation_expr(
                cell_act,
                paste0("(", h_inp[h], " + ", coupled, ")")
              )
            } else {
              # Legacy reset_after=FALSE formula: (r ⊙ H_prev) @ Rh
              coupled <- paste(
                paste0(
                  backtick(r_nms),
                  " * ",
                  backtick(H_prev_nms),
                  " * ",
                  format_numeric(R_h[h, ])
                ),
                collapse = " + "
              )
              activation_expr(
                cell_act,
                paste0("(", h_inp[h], " + ", coupled, ")")
              )
            }
          },
          character(1L)
        )
        h_tilde_nms <- paste0(
          "orbital_gru_",
          lname,
          "_ht_t",
          t,
          "_h",
          seq_len(H)
        )

        # H_t = (1 - z_t) * h̃_t + z_t * H_prev
        H_cur_nms <- paste0("orbital_gru_", lname, "_H_t", t, "_h", seq_len(H))
        H_cur_exprs <- if (is.null(H_prev_nms)) {
          paste0("((1 - ", backtick(z_nms), ") * ", backtick(h_tilde_nms), ")")
        } else {
          paste0(
            "((1 - ",
            backtick(z_nms),
            ") * ",
            backtick(h_tilde_nms),
            " + ",
            backtick(z_nms),
            " * ",
            backtick(H_prev_nms),
            ")"
          )
        }

        # Order matters for SQL column dependencies: z, r, h̃, H
        all_gru_nms <- c(all_gru_nms, z_nms, r_nms, h_tilde_nms, H_cur_nms)
        all_gru_exprs <- c(
          all_gru_exprs,
          z_exprs,
          r_exprs,
          h_tilde_exprs,
          H_cur_exprs
        )
        all_H_nms[[t]] <- H_cur_nms

        H_prev_nms <- H_cur_nms
      }

      all_exprs[[lname]] <- stats::setNames(all_gru_exprs, all_gru_nms)
      out_nms <- if (return_seq) {
        unlist(all_H_nms, use.names = FALSE)
      } else {
        H_prev_nms
      }
      assign(lname, out_nms, envir = expr_reg)
    } else if (grepl("bidirectional", cls)) {
      # Bidirectional wrapper: runs forward and backward passes of an inner
      # LSTM or GRU and concatenates their outputs.
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

      fwd_layer <- tryCatch(l$forward_layer, error = function(e) NULL)
      bwd_layer <- tryCatch(l$backward_layer, error = function(e) NULL)
      if (is.null(fwd_layer) || is.null(bwd_layer)) {
        cli::cli_abort(
          c(
            "Bidirectional layer {.val {lname}} does not expose forward_layer / backward_layer.",
            "i" = "Ensure you are using Keras 3 (>= 3.0)."
          )
        )
      }
      fwd_cls_inner <- tolower(class(fwd_layer)[1L])
      if (!grepl("\\blstm\\b|\\bgru\\b", fwd_cls_inner, perl = TRUE)) {
        cli::cli_abort(
          "Bidirectional layer {.val {lname}}: inner layer {.cls {fwd_cls_inner}} is not LSTM or GRU."
        )
      }

      cfg_l <- tryCatch(l$get_config(), error = function(e) list())
      merge_mode <- tryCatch(
        tolower(as.character(cfg_l$merge_mode)),
        error = function(e) "concat"
      )
      if (is.na(merge_mode) || merge_mode == "null" || !nzchar(merge_mode)) {
        merge_mode <- "concat"
      }
      if (!merge_mode %in% c("concat", "sum", "mul", "ave")) {
        cli::cli_abort(
          "Bidirectional layer {.val {lname}}: merge_mode = {.val {merge_mode}} is not supported."
        )
      }

      # Infer I_feat and T_len from forward layer kernel shape
      fwd_wts <- fwd_layer$get_weights()
      I_feat_bd <- nrow(fwd_wts[[1L]])
      T_len_bd <- as.integer(length(in_exprs) / I_feat_bd)

      fwd_pfx <- paste0("orbital_bidir_fwd_", lname)
      fwd <- .unroll_rnn(fwd_layer, in_exprs, fwd_pfx)

      rev_exprs <- unlist(
        lapply(rev(seq_len(T_len_bd)), function(t) {
          in_exprs[((t - 1L) * I_feat_bd + 1L):(t * I_feat_bd)]
        }),
        use.names = FALSE
      )
      bwd_pfx <- paste0("orbital_bidir_bwd_", lname)
      bwd <- .unroll_rnn(bwd_layer, rev_exprs, bwd_pfx)

      all_exprs[[lname]] <- stats::setNames(
        c(fwd$exprs, bwd$exprs),
        c(fwd$nms, bwd$nms)
      )

      # Determine the inner layer's return_sequences flag
      inner_return_seq <- isTRUE(
        tryCatch(
          as.logical(fwd_layer$get_config()$return_sequences),
          error = function(e) FALSE
        )
      )

      if (merge_mode == "concat") {
        out_nms_bd <- if (inner_return_seq) {
          # Per timestep t: [fwd[t] || bwd[T-t+1]]
          # (bwd processed reversed input; bwd step k = original step T-k+1)
          unlist(
            lapply(seq_len(fwd$T), function(t) {
              c(fwd$all_H_nms[[t]], bwd$all_H_nms[[fwd$T - t + 1L]])
            }),
            use.names = FALSE
          )
        } else {
          c(fwd$out_nms, bwd$out_nms)
        }
      } else {
        # sum / mul / ave: emit merged intermediate expressions
        fwd_final <- if (inner_return_seq) {
          unlist(fwd$all_H_nms, use.names = FALSE)
        } else {
          fwd$out_nms
        }
        bwd_final <- if (inner_return_seq) {
          unlist(
            lapply(seq_len(fwd$T), function(t) {
              bwd$all_H_nms[[fwd$T - t + 1L]]
            }),
            use.names = FALSE
          )
        } else {
          bwd$out_nms
        }
        merge_nms <- paste0(
          "orbital_bidir_merge_",
          lname,
          "_",
          seq_along(fwd_final)
        )
        merge_exprs_bd <- vapply(
          seq_along(fwd_final),
          function(i) {
            switch(
              merge_mode,
              sum = paste0(
                "(",
                backtick(fwd_final[i]),
                " + ",
                backtick(bwd_final[i]),
                ")"
              ),
              mul = paste0(
                "(",
                backtick(fwd_final[i]),
                " * ",
                backtick(bwd_final[i]),
                ")"
              ),
              ave = paste0(
                "((",
                backtick(fwd_final[i]),
                " + ",
                backtick(bwd_final[i]),
                ") / 2)"
              )
            )
          },
          character(1L)
        )
        all_exprs[[lname]] <- c(
          all_exprs[[lname]],
          stats::setNames(merge_exprs_bd, merge_nms)
        )
        out_nms_bd <- merge_nms
      }
      assign(lname, out_nms_bd, envir = expr_reg)
    } else if (grepl("simplernn", cls)) {
      # SimpleRNN: h_t = activation(x_t @ W + h_{t-1} @ U + b)
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      wts <- l$get_weights() # kernel (I, H), recurrent_kernel (H, H) [, bias (H,)]
      kernel <- wts[[1L]] # (I, H)
      rkernel <- wts[[2L]] # (H, H)
      H <- ncol(kernel)
      I_feat <- nrow(kernel)
      T_len <- as.integer(length(in_exprs) / I_feat)
      bias_v <- if (length(wts) >= 3L) as.numeric(wts[[3L]]) else numeric(H)
      cfg_l <- tryCatch(l$get_config(), error = function(e) list())
      activation <- tryCatch(
        tolower(as.character(cfg_l$activation %||% "tanh")),
        error = function(e) "tanh"
      )
      return_seq <- isTRUE(
        tryCatch(as.logical(cfg_l$return_sequences), error = function(e) FALSE)
      )
      W_t <- t(kernel) # (H, I)
      U_t <- t(rkernel) # (H, H)

      all_srnn_nms <- character(0L)
      all_srnn_exprs <- character(0L)
      all_H_nms <- list()
      H_prev_nms <- NULL

      for (t in seq_len(T_len)) {
        x_t_exprs <- in_exprs[((t - 1L) * I_feat + 1L):(t * I_feat)]
        pre_act <- build_mlp_pre_act(W_t, bias_v, x_t_exprs)
        if (!is.null(H_prev_nms)) {
          rec <- build_mlp_pre_act(U_t, numeric(H), H_prev_nms)
          pre_act <- paste0("(", pre_act, " + ", rec, ")")
        }
        H_nms <- paste0("orbital_srnn_", lname, "_H_t", t, "_h", seq_len(H))
        H_exprs_t <- vapply(
          pre_act,
          function(e) activation_expr(activation, e),
          character(1L)
        )
        all_srnn_nms <- c(all_srnn_nms, H_nms)
        all_srnn_exprs <- c(all_srnn_exprs, H_exprs_t)
        all_H_nms[[t]] <- H_nms
        H_prev_nms <- H_nms
      }
      all_exprs[[lname]] <- stats::setNames(all_srnn_exprs, all_srnn_nms)
      out_nms <- if (return_seq) {
        unlist(all_H_nms, use.names = FALSE)
      } else {
        H_prev_nms
      }
      assign(lname, out_nms, envir = expr_reg)
    } else if (grepl("unitnorm", cls)) {
      # UnitNormalization: L2-normalise each row across all features.
      # y_i = x_i / sqrt(x_1^2 + ... + x_n^2 + eps)
      # Only axis = -1 (normalise over feature dimension) is supported.
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      n_feat <- length(in_exprs)
      expr_bt <- backtick(in_exprs)
      norm_nm <- paste0("orbital_unitnorm_", lname, "_norm")
      sq_sum <- paste(paste0(expr_bt, "^2"), collapse = " + ")
      norm_expr <- paste0("sqrt(", sq_sum, " + 1e-12)")
      unit_names <- paste0("orbital_unitnorm_", lname, "_h", seq_len(n_feat))
      unit_exprs <- vapply(
        seq_len(n_feat),
        function(i) paste0("(", expr_bt[i], " / `", norm_nm, "`)"),
        character(1L)
      )
      all_exprs[[lname]] <- c(
        stats::setNames(norm_expr, norm_nm),
        stats::setNames(unit_exprs, unit_names)
      )
      assign(lname, unit_names, envir = expr_reg)
    } else if (grepl("zeropadding1d", cls)) {
      # ZeroPadding1D: emit literal 0 columns for padded timesteps;
      # pass through interior columns unchanged.
      # Input layout: [T_in × C] flat vector (time-step major).
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

      padding_raw <- tryCatch(l$get_config()$padding, error = function(e) 1L)
      if (is.list(padding_raw)) {
        padding_raw <- as.integer(unlist(padding_raw))
      } else {
        padding_raw <- as.integer(padding_raw)
      }
      pad_left <- if (length(padding_raw) >= 1L) padding_raw[[1L]] else 1L
      pad_right <- if (length(padding_raw) >= 2L) {
        padding_raw[[2L]]
      } else {
        padding_raw[[1L]]
      }

      # Determine C (channels per timestep) from input shape; fallback to 1.
      in_shape <- tryCatch(
        as.integer(unlist(l$input_shape)),
        error = function(e) NULL
      )
      C_feat <- if (!is.null(in_shape) && length(in_shape) >= 1L) {
        tail(in_shape[!is.na(in_shape)], 1L)
      } else {
        1L
      }
      T_in <- as.integer(length(in_exprs) / C_feat)
      T_out <- T_in + pad_left + pad_right

      zero_col <- "0"
      pad_block <- rep(zero_col, C_feat)

      out_exprs <- character(0L)
      out_nms <- character(0L)
      for (tp in seq_len(T_out)) {
        step_nms <- paste0(
          "orbital_zpad_",
          lname,
          "_t",
          tp,
          "_c",
          seq_len(C_feat)
        )
        if (tp <= pad_left || tp > pad_left + T_in) {
          step_exprs <- pad_block
        } else {
          orig_t <- tp - pad_left
          step_exprs <- in_exprs[
            ((orig_t - 1L) * C_feat + 1L):(orig_t * C_feat)
          ]
        }
        out_exprs <- c(out_exprs, step_exprs)
        out_nms <- c(out_nms, step_nms)
      }
      all_exprs[[lname]] <- stats::setNames(out_exprs, out_nms)
      assign(lname, out_nms, envir = expr_reg)
    } else {
      cli::cli_abort(c(
        "Unsupported layer type in Keras Functional model: {.cls {cls_orig}}.",
        "i" = paste(
          "orbital supports: Dense, Add, Concatenate, BatchNormalization,",
          "LayerNormalization, InstanceNormalization, GroupNormalization,",
          "RMSNormalization, PReLU, GlobalAveragePooling1D, GlobalMaxPooling1D,",
          "AveragePooling1D, MaxPooling1D, GlobalSumPooling1D,",
          "Conv1D, LSTM, GRU, Bidirectional(LSTM/GRU), SimpleRNN,",
          "UnitNormalization, ZeroPadding1D,",
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
        paste0(
          "\\badd\\b|concatenate|batchnorm|layernorm|instancenorm|groupnorm|",
          "rmsnormalization|prelu|leakyrelu|\\belu\\b|globalaveragepool|",
          "globalmaxpool|averagepooling1d|maxpooling1d|globalsumpooling|",
          "\\bactivation\\b|\\bsoftmax\\b|\\blstm\\b|\\bgru\\b|conv1d|",
          "bidirectional|simplernn|unitnorm|zeropadding1d"
        ),
        cls,
        perl = TRUE
      ) &&
        !grepl(
          "dense|input|flatten|reshape|dropout|depthwise|separable|2d|3d",
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
