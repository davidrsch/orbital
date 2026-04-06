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
    } else if (grepl("einsumdense", cls)) {
      # EinsumDense ─ generalised einsum projection; only Dense-equivalent equations.
      # Supported:
      #   "ab,bc->ac"   : input (B,A), kernel (A,C), output (B,C)  — simple Dense
      #   "...b,bc->...c": same with NumPy ellipsis                 — simple Dense
      #   "abc,cd->abd" : input (B,T,C), kernel (C,D), output (B,T,D) — time-dist Dense
      inbound <- topo_map[[lname]]
      if (is.null(inbound) || length(inbound) < 1L) {
        cli::cli_abort(
          "EinsumDense layer {.val {lname}} has no inbound connections in model config."
        )
      }
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      wts <- l$get_weights()
      if (length(wts) < 1L) {
        cli::cli_abort("EinsumDense layer {.val {lname}} has no weights.")
      }
      cfg_l <- tryCatch(l$get_config(), error = function(e) list())
      equation_raw <- tryCatch(as.character(cfg_l$equation), error = function(e) "")
      equation <- gsub("\\s+", "", equation_raw)

      activation_config <- tryCatch(cfg_l$activation, error = function(e) NULL)
      activation <- if (is.null(activation_config)) {
        "linear"
      } else if (is.list(activation_config)) {
        tolower(as.character(activation_config$class_name)[1L])
      } else {
        tolower(as.character(activation_config)[1L])
      }
      if (!nzchar(activation)) activation <- "linear"

      kern <- wts[[1L]]
      bias_v_raw <- if (length(wts) >= 2L) as.numeric(wts[[2L]]) else NULL

      if (equation %in% c("ab,bc->ac", "...b,bc->...c")) {
        # Simple Dense: kernel (A, C); treat like Dense.
        n_out <- ncol(kern)
        bias_use <- if (is.null(bias_v_raw)) numeric(n_out) else bias_v_raw
        pre_act <- build_mlp_pre_act(t(kern), bias_use, in_exprs)
        act_exprs <- vapply(
          pre_act,
          function(z) activation_expr(activation, z),
          character(1L)
        )
        unit_names <- paste0(
          "orbital_einsumdense_", lname, "_h", seq_along(act_exprs)
        )
        all_exprs[[lname]] <- stats::setNames(act_exprs, unit_names)
        assign(lname, unit_names, envir = expr_reg)
      } else if (equation == "abc,cd->abd") {
        # Time-distributed Dense: kernel (C, D); apply per spatial position.
        C_last <- nrow(kern)
        D_out <- ncol(kern)
        T_len <- as.integer(length(in_exprs) / C_last)
        bias_use <- if (is.null(bias_v_raw)) numeric(D_out) else bias_v_raw
        k_T <- t(kern) # (D_out, C_last) for build_mlp_pre_act
        out_nms <- character(0L)
        out_exprs <- character(0L)
        for (p in seq_len(T_len)) {
          in_slice <- in_exprs[((p - 1L) * C_last + 1L):(p * C_last)]
          pre_act <- build_mlp_pre_act(k_T, bias_use, in_slice)
          ae <- vapply(
            pre_act,
            function(z) activation_expr(activation, z),
            character(1L)
          )
          nms <- paste0(
            "orbital_einsumdense_", lname, "_t", p, "_h", seq_len(D_out)
          )
          out_nms <- c(out_nms, nms)
          out_exprs <- c(out_exprs, ae)
        }
        all_exprs[[lname]] <- stats::setNames(out_exprs, out_nms)
        assign(lname, out_nms, envir = expr_reg)
      } else {
        cli::cli_abort(c(
          "EinsumDense layer {.val {lname}}: equation {.val {equation}} is not supported by orbital.",
          "i" = "Supported equations: 'ab,bc->ac', '...b,bc->...c', 'abc,cd->abd'."
        ))
      }
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
    } else if (grepl("\\bmultiply\\b", cls, perl = TRUE)) {
      # Element-wise Multiply: element-wise product of >=2 inputs
      inbound <- topo_map[[lname]]
      if (is.null(inbound) || length(inbound) < 2L) {
        cli::cli_abort(
          "Keras Multiply layer {.val {lname}} must have at least 2 inbound inputs, got {length(inbound)}."
        )
      }
      all_inbound_exprs <- lapply(inbound, function(nm) {
        get(nm, envir = expr_reg, inherits = FALSE)
      })
      widths <- lengths(all_inbound_exprs)
      if (length(unique(widths)) != 1L) {
        cli::cli_abort(
          "Keras Multiply layer {.val {lname}}: all inputs must have the same width (got: {paste(widths, collapse = ', ')})."
        )
      }
      mul_names <- paste0("orbital_", lname, "_h", seq_len(widths[1L]))
      mul_exprs <- vapply(
        seq_len(widths[1L]),
        function(i) {
          terms <- vapply(
            all_inbound_exprs,
            function(e) backtick(e[i]),
            character(1L)
          )
          paste0("(", paste(terms, collapse = " * "), ")")
        },
        character(1L)
      )
      all_exprs[[lname]] <- stats::setNames(mul_exprs, mul_names)
      assign(lname, mul_names, envir = expr_reg)
    } else if (
      grepl("\\baverage\\b", cls, perl = TRUE) && !grepl("pool|global", cls)
    ) {
      # Element-wise Average: mean of >=2 inputs
      inbound <- topo_map[[lname]]
      if (is.null(inbound) || length(inbound) < 2L) {
        cli::cli_abort(
          "Keras Average layer {.val {lname}} must have at least 2 inbound inputs, got {length(inbound)}."
        )
      }
      all_inbound_exprs <- lapply(inbound, function(nm) {
        get(nm, envir = expr_reg, inherits = FALSE)
      })
      widths <- lengths(all_inbound_exprs)
      if (length(unique(widths)) != 1L) {
        cli::cli_abort(
          "Keras Average layer {.val {lname}}: all inputs must have the same width (got: {paste(widths, collapse = ', ')})."
        )
      }
      n_inputs <- length(inbound)
      avg_names <- paste0("orbital_", lname, "_h", seq_len(widths[1L]))
      avg_exprs <- vapply(
        seq_len(widths[1L]),
        function(i) {
          terms <- vapply(
            all_inbound_exprs,
            function(e) backtick(e[i]),
            character(1L)
          )
          paste0(
            "((",
            paste(terms, collapse = " + "),
            ") / ",
            format_numeric(n_inputs),
            ")"
          )
        },
        character(1L)
      )
      all_exprs[[lname]] <- stats::setNames(avg_exprs, avg_names)
      assign(lname, avg_names, envir = expr_reg)
    } else if (grepl("\\bmaximum\\b", cls, perl = TRUE)) {
      # Element-wise Maximum: per-element max over >=2 inputs
      inbound <- topo_map[[lname]]
      if (is.null(inbound) || length(inbound) < 2L) {
        cli::cli_abort(
          "Keras Maximum layer {.val {lname}} must have at least 2 inbound inputs, got {length(inbound)}."
        )
      }
      all_inbound_exprs <- lapply(inbound, function(nm) {
        get(nm, envir = expr_reg, inherits = FALSE)
      })
      widths <- lengths(all_inbound_exprs)
      if (length(unique(widths)) != 1L) {
        cli::cli_abort(
          "Keras Maximum layer {.val {lname}}: all inputs must have the same width (got: {paste(widths, collapse = ', ')})."
        )
      }
      max_names <- paste0("orbital_", lname, "_h", seq_len(widths[1L]))
      max_exprs <- vapply(
        seq_len(widths[1L]),
        function(i) {
          terms <- vapply(
            all_inbound_exprs,
            function(e) backtick(e[i]),
            character(1L)
          )
          paste0("pmax(", paste(terms, collapse = ", "), ")")
        },
        character(1L)
      )
      all_exprs[[lname]] <- stats::setNames(max_exprs, max_names)
      assign(lname, max_names, envir = expr_reg)
    } else if (grepl("\\bminimum\\b", cls, perl = TRUE)) {
      # Element-wise Minimum: per-element min over >=2 inputs
      inbound <- topo_map[[lname]]
      if (is.null(inbound) || length(inbound) < 2L) {
        cli::cli_abort(
          "Keras Minimum layer {.val {lname}} must have at least 2 inbound inputs, got {length(inbound)}."
        )
      }
      all_inbound_exprs <- lapply(inbound, function(nm) {
        get(nm, envir = expr_reg, inherits = FALSE)
      })
      widths <- lengths(all_inbound_exprs)
      if (length(unique(widths)) != 1L) {
        cli::cli_abort(
          "Keras Minimum layer {.val {lname}}: all inputs must have the same width (got: {paste(widths, collapse = ', ')})."
        )
      }
      min_names <- paste0("orbital_", lname, "_h", seq_len(widths[1L]))
      min_exprs <- vapply(
        seq_len(widths[1L]),
        function(i) {
          terms <- vapply(
            all_inbound_exprs,
            function(e) backtick(e[i]),
            character(1L)
          )
          paste0("pmin(", paste(terms, collapse = ", "), ")")
        },
        character(1L)
      )
      all_exprs[[lname]] <- stats::setNames(min_exprs, min_names)
      assign(lname, min_names, envir = expr_reg)
    } else if (grepl("\\bsubtract\\b", cls, perl = TRUE)) {
      # Element-wise Subtract: first_input - second_input
      inbound <- topo_map[[lname]]
      if (is.null(inbound) || length(inbound) != 2L) {
        cli::cli_abort(
          "Keras Subtract layer {.val {lname}} requires exactly 2 inbound inputs, got {length(inbound)}."
        )
      }
      exprs_a <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      exprs_b <- get(inbound[2L], envir = expr_reg, inherits = FALSE)
      if (length(exprs_a) != length(exprs_b)) {
        cli::cli_abort(
          "Keras Subtract layer {.val {lname}}: both inputs must have the same width (got: {length(exprs_a)} vs {length(exprs_b)})."
        )
      }
      sub_names <- paste0("orbital_", lname, "_h", seq_len(length(exprs_a)))
      sub_exprs <- vapply(
        seq_along(exprs_a),
        function(i) paste0(backtick(exprs_a[i]), " - ", backtick(exprs_b[i])),
        character(1L)
      )
      all_exprs[[lname]] <- stats::setNames(sub_exprs, sub_names)
      assign(lname, sub_names, envir = expr_reg)
    } else if (grepl("\\bdot\\b", cls, perl = TRUE)) {
      # Dot product: sum of element-wise products of exactly 2 inputs (axes=-1)
      inbound <- topo_map[[lname]]
      if (is.null(inbound) || length(inbound) != 2L) {
        cli::cli_abort(
          "Keras Dot layer {.val {lname}} requires exactly 2 inbound inputs, got {length(inbound)}."
        )
      }
      exprs_a <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      exprs_b <- get(inbound[2L], envir = expr_reg, inherits = FALSE)
      if (length(exprs_a) != length(exprs_b)) {
        cli::cli_abort(
          "Keras Dot layer {.val {lname}}: both inputs must have the same width."
        )
      }
      cfg <- tryCatch(l$get_config(), error = function(e) list())
      axes <- tryCatch(as.integer(cfg$axes), error = function(e) -1L)
      if (!all(axes %in% c(-1L, 1L))) {
        cli::cli_abort(
          "Keras Dot layer {.val {lname}}: only axes=-1 (feature-axis dot product) is supported."
        )
      }
      dot_name <- paste0("orbital_", lname, "_dot")
      terms <- vapply(
        seq_along(exprs_a),
        function(i) {
          paste0("(", backtick(exprs_a[i]), " * ", backtick(exprs_b[i]), ")")
        },
        character(1L)
      )
      dot_expr <- paste0("(", paste(terms, collapse = " + "), ")")
      all_exprs[[lname]] <- stats::setNames(dot_expr, dot_name)
      assign(lname, dot_name, envir = expr_reg)
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
    } else if (
      grepl("\\bactivation\\b", cls, perl = TRUE) && !grepl("softmax", cls)
    ) {
      # Standalone Activation layer: apply activation function to inbound expressions
      # NOTE: must be checked BEFORE the generic softmax branch is unnecessary —
      # the !grepl("softmax") guard prevents Keras3 Softmax() layers (whose class
      # path contains "activation") from being silently treated as linear identity.
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
    } else if (grepl("adaptiveaveragepooling1d", cls)) {
      # AdaptiveAveragePooling1D: adaptive-window mean.
      # For output size O and input length T_in, output position i (0-indexed):
      #   start = floor(i * T_in / O), end = ceiling((i+1) * T_in / O)
      #   output = mean over that window.
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      n_f <- length(in_exprs)
      o_size <- tryCatch(
        as.integer(l$get_config()$output_size),
        error = function(e) NA_integer_
      )
      if (is.na(o_size) || o_size < 1L) {
        cli::cli_abort(
          "AdaptiveAveragePooling1D: invalid output_size {.val {o_size}}."
        )
      }
      in_shape <- tryCatch(
        as.integer(unlist(l$input_shape)),
        error = function(e) NULL
      )
      C_feat <- if (!is.null(in_shape) && length(in_shape) >= 1L) {
        tail(in_shape[!is.na(in_shape)], 1L)
      } else {
        1L
      }
      T_in <- as.integer(n_f / C_feat)
      if (T_in < o_size) {
        cli::cli_abort(
          "AdaptiveAveragePooling1D: output_size ({o_size}) must not exceed input length ({T_in})."
        )
      }
      pool_nms <- character(0L)
      pool_exprs <- character(0L)
      idx <- 1L
      for (i in seq_len(o_size) - 1L) {
        start_t <- as.integer(floor(i * T_in / o_size))
        end_t <- as.integer(ceiling((i + 1L) * T_in / o_size))
        for (c in seq_len(C_feat)) {
          cols_bt <- vapply(
            seq(start_t, end_t - 1L),
            function(t) backtick(in_exprs[[t * C_feat + c]]),
            character(1L)
          )
          n_win <- length(cols_bt)
          nm <- paste0("orbital_adaptiveavgpool1d_", lname, "_", idx)
          pool_exprs[[idx]] <- paste0(
            "(",
            paste(cols_bt, collapse = " + "),
            ") / ",
            n_win
          )
          pool_nms[[idx]] <- nm
          idx <- idx + 1L
        }
      }
      all_exprs[[lname]] <- stats::setNames(pool_exprs, pool_nms)
      assign(lname, pool_nms, envir = expr_reg)
    } else if (grepl("averagepooling1d", cls)) {
      # AveragePooling1D: sliding-window mean across time steps.
      # Input layout: time-step major (T_in × C_feat) — in_exprs[(t-1)*C_feat + c]
      # for 1-indexed timestep t and 1-indexed channel c.
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
      if (is.na(pool_size)) {
        pool_size <- n_f
      }
      stride <- tryCatch(
        {
          s <- as.integer(unlist(l$get_config()$strides))[[1L]]
          if (!is.na(s)) s else pool_size
        },
        error = function(e) pool_size
      )
      padding <- tryCatch(
        tolower(as.character(l$get_config()$padding)),
        error = function(e) "valid"
      )
      in_shape <- tryCatch(
        as.integer(unlist(l$input_shape)),
        error = function(e) NULL
      )
      C_feat <- if (!is.null(in_shape) && length(in_shape) >= 1L) {
        tail(in_shape[!is.na(in_shape)], 1L)
      } else {
        1L
      }
      T_in <- as.integer(n_f / C_feat)
      if (padding == "valid") {
        T_out <- (T_in - pool_size) %/% stride + 1L
        pad_l <- 0L
      } else {
        T_out <- as.integer(ceiling(T_in / stride))
        total_pad <- max(0L, (T_out - 1L) * stride + pool_size - T_in)
        pad_l <- total_pad %/% 2L
      }
      pool_nms <- character(0)
      pool_exprs <- character(0)
      idx <- 1L
      for (p in seq_len(T_out) - 1L) {
        for (c in seq_len(C_feat)) {
          window_bt <- character(0)
          for (kk in seq_len(pool_size) - 1L) {
            t_pos <- p * stride + kk - pad_l
            if (t_pos >= 0L && t_pos < T_in) {
              flat_idx <- t_pos * C_feat + c
              window_bt <- c(window_bt, backtick(in_exprs[[flat_idx]]))
            }
          }
          n_valid <- length(window_bt)
          nm <- paste0("orbital_avgpool1d_", lname, "_", idx)
          # For "same" padding, Keras 3 divides by pool_size (including
          # implicit zero-padded positions), not by n_valid (in-bounds only).
          # For "valid" padding all windows are full so pool_size == n_valid.
          denom <- if (padding == "same") pool_size else n_valid
          pool_exprs[[idx]] <- if (n_valid > 0L) {
            paste0("(", paste(window_bt, collapse = " + "), ") / ", denom)
          } else {
            "0"
          }
          pool_nms[[idx]] <- nm
          idx <- idx + 1L
        }
      }
      all_exprs[[lname]] <- stats::setNames(pool_exprs, pool_nms)
      assign(lname, pool_nms, envir = expr_reg)
    } else if (grepl("adaptivemaxpooling1d", cls)) {
      # AdaptiveMaxPooling1D: adaptive-window max.
      # Same window formula as AdaptiveAveragePooling1D, max instead of mean.
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      n_f <- length(in_exprs)
      o_size <- tryCatch(
        as.integer(l$get_config()$output_size),
        error = function(e) NA_integer_
      )
      if (is.na(o_size) || o_size < 1L) {
        cli::cli_abort(
          "AdaptiveMaxPooling1D: invalid output_size {.val {o_size}}."
        )
      }
      in_shape <- tryCatch(
        as.integer(unlist(l$input_shape)),
        error = function(e) NULL
      )
      C_feat <- if (!is.null(in_shape) && length(in_shape) >= 1L) {
        tail(in_shape[!is.na(in_shape)], 1L)
      } else {
        1L
      }
      T_in <- as.integer(n_f / C_feat)
      if (T_in < o_size) {
        cli::cli_abort(
          "AdaptiveMaxPooling1D: output_size ({o_size}) must not exceed input length ({T_in})."
        )
      }
      pool_nms <- character(0L)
      pool_exprs <- character(0L)
      idx <- 1L
      for (i in seq_len(o_size) - 1L) {
        start_t <- as.integer(floor(i * T_in / o_size))
        end_t <- as.integer(ceiling((i + 1L) * T_in / o_size))
        for (c in seq_len(C_feat)) {
          cols_bt <- vapply(
            seq(start_t, end_t - 1L),
            function(t) backtick(in_exprs[[t * C_feat + c]]),
            character(1L)
          )
          nm <- paste0("orbital_adaptivemaxpool1d_", lname, "_", idx)
          pool_exprs[[idx]] <- paste0(
            "do.call(pmax, list(",
            paste(cols_bt, collapse = ", "),
            "))"
          )
          pool_nms[[idx]] <- nm
          idx <- idx + 1L
        }
      }
      all_exprs[[lname]] <- stats::setNames(pool_exprs, pool_nms)
      assign(lname, pool_nms, envir = expr_reg)
    } else if (grepl("maxpooling1d", cls)) {
      # MaxPooling1D: sliding-window max across time steps.
      # Input layout: time-step major (T_in × C_feat) — in_exprs[(t-1)*C_feat + c]
      # for 1-indexed timestep t and 1-indexed channel c.
      if (grepl("2d|3d", cls)) {
        cli::cli_abort(
          "MaxPooling2D/3D is not supported by orbital (requires spatial aggregation)."
        )
      }
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      n_f <- length(in_exprs)
      pool_size <- tryCatch(
        as.integer(l$get_config()$pool_size),
        error = function(e) NA_integer_
      )
      if (is.na(pool_size)) {
        pool_size <- n_f
      }
      stride <- tryCatch(
        {
          s <- as.integer(unlist(l$get_config()$strides))[[1L]]
          if (!is.na(s)) s else pool_size
        },
        error = function(e) pool_size
      )
      padding <- tryCatch(
        tolower(as.character(l$get_config()$padding)),
        error = function(e) "valid"
      )
      in_shape <- tryCatch(
        as.integer(unlist(l$input_shape)),
        error = function(e) NULL
      )
      C_feat <- if (!is.null(in_shape) && length(in_shape) >= 1L) {
        tail(in_shape[!is.na(in_shape)], 1L)
      } else {
        1L
      }
      T_in <- as.integer(n_f / C_feat)
      if (padding == "valid") {
        T_out <- (T_in - pool_size) %/% stride + 1L
        pad_l <- 0L
      } else {
        T_out <- as.integer(ceiling(T_in / stride))
        total_pad <- max(0L, (T_out - 1L) * stride + pool_size - T_in)
        pad_l <- total_pad %/% 2L
      }
      pool_nms <- character(0)
      pool_exprs <- character(0)
      idx <- 1L
      for (p in seq_len(T_out) - 1L) {
        for (c in seq_len(C_feat)) {
          window_bt <- character(0)
          for (kk in seq_len(pool_size) - 1L) {
            t_pos <- p * stride + kk - pad_l
            if (t_pos >= 0L && t_pos < T_in) {
              flat_idx <- t_pos * C_feat + c
              window_bt <- c(window_bt, backtick(in_exprs[[flat_idx]]))
            }
          }
          nm <- paste0("orbital_maxpool1d_", lname, "_", idx)
          pool_exprs[[idx]] <- if (length(window_bt) > 0L) {
            paste0(
              "do.call(pmax, list(",
              paste(window_bt, collapse = ", "),
              "))"
            )
          } else {
            "-Inf"
          }
          pool_nms[[idx]] <- nm
          idx <- idx + 1L
        }
      }
      all_exprs[[lname]] <- stats::setNames(pool_exprs, pool_nms)
      assign(lname, pool_nms, envir = expr_reg)
    } else if (grepl("upsampling1d", cls)) {
      # UpSampling1D: repeat each time-step `size` times.
      # Input layout: time-step major (T_in × C_feat).
      # Output: T_out = T_in * size timesteps, same C_feat channels.
      if (grepl("2d|3d", cls)) {
        cli::cli_abort(
          "UpSampling2D/3D is not supported by orbital (requires spatial replication)."
        )
      }
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      n_f <- length(in_exprs)
      size <- tryCatch(
        as.integer(l$get_config()$size),
        error = function(e) NA_integer_
      )
      if (is.na(size) || size < 1L) {
        size <- 2L
      }
      in_shape <- tryCatch(
        as.integer(unlist(l$input_shape)),
        error = function(e) NULL
      )
      C_feat <- if (!is.null(in_shape) && length(in_shape) >= 1L) {
        tail(in_shape[!is.na(in_shape)], 1L)
      } else {
        1L
      }
      T_in <- as.integer(n_f / C_feat)
      T_out <- T_in * size
      up_nms <- character(T_out * C_feat)
      up_exprs <- character(T_out * C_feat)
      idx <- 1L
      for (p in seq_len(T_out) - 1L) {
        t_src <- p %/% size
        for (c in seq_len(C_feat)) {
          src_idx <- t_src * C_feat + c
          up_nms[[idx]] <- paste0("orbital_up1d_", lname, "_", idx)
          up_exprs[[idx]] <- backtick(in_exprs[[src_idx]])
          idx <- idx + 1L
        }
      }
      all_exprs[[lname]] <- stats::setNames(up_exprs, up_nms)
      assign(lname, up_nms, envir = expr_reg)
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
    } else if (grepl("conv1dtranspose", cls)) {
      # Conv1DTranspose ─ transposed (fractionally-strided) 1-D convolution.
      # Keras weight layout (same as Conv1D):
      #   kernel : (kernel_size, in_channels, filters)  i.e. (kW, C_in, C_out)
      #   bias   : (filters,)  [optional]
      # Dilations > 1 are not supported (raise cli_abort).
      # Padding: "valid" or "same".
      # W_out formula:
      #   valid: W_out = (W_in - 1) * stride + kW
      #   same:  W_out = W_in * stride
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

      if (dilation > 1L) {
        cli::cli_abort(
          "Keras Conv1DTranspose layer {.val {lname}}: dilation_rate > 1 is not supported."
        )
      }

      n_total <- length(in_exprs)
      w_in <- as.integer(n_total / c_in)

      if (pad_type == "same") {
        w_out <- w_in * stride
        pad_total <- max(0L, k_w - stride)
        pad_l <- as.integer(floor(pad_total / 2L))
      } else {
        # "valid"
        w_out <- (w_in - 1L) * stride + k_w
        pad_l <- 0L
      }

      tconv_nms <- character(0L)
      tconv_exprs <- character(0L)

      # For each output position q (0-indexed) and output channel f,
      # gather all input (p, k, c) triples that contribute:
      #   out[q, f] = sum_{p,k,c} in[p, c] * kern[k, c, f]
      # where p * stride + k - pad_l == q  (dilation=1)
      for (q in seq_len(w_out)) {
        q0 <- q - 1L
        for (f in seq_len(c_out)) {
          terms <- character(0L)
          for (k in seq_len(k_w)) {
            k0 <- k - 1L
            # p * stride = q0 + pad_l - k0
            num <- q0 + pad_l - k0
            if (num >= 0L && num %% stride == 0L) {
              p0 <- as.integer(num / stride)
              if (p0 >= 0L && p0 < w_in) {
                for (c in seq_len(c_in)) {
                  feat_nm <- in_exprs[p0 * c_in + c]
                  wt_val <- format_numeric(kern[k, c, f])
                  terms <- c(
                    terms,
                    paste0("(", backtick(feat_nm), " * ", wt_val, ")")
                  )
                }
              }
            }
          }
          b_str <- format_numeric(bias_v[f])
          expr_str <- if (length(terms) == 0L) {
            b_str
          } else {
            paste0("(", paste(c(terms, b_str), collapse = " + "), ")")
          }
          nm <- paste0("orbital_tconv_", lname, "_q", q, "_f", f)
          tconv_nms <- c(tconv_nms, nm)
          tconv_exprs <- c(tconv_exprs, expr_str)
        }
      }
      all_exprs[[lname]] <- stats::setNames(tconv_exprs, tconv_nms)
      assign(lname, tconv_nms, envir = expr_reg)
    } else if (grepl("depthwiseconv1d", cls)) {
      # DepthwiseConv1D ─ channel-wise 1-D convolution (no cross-channel mixing).
      # Keras weight layout:
      #   depthwise_kernel : (kernel_size, in_channels, depth_multiplier)
      #   bias             : (in_channels * depth_multiplier,)  [optional]
      # orbital output convention: T_out × (C_in * depth_mult) columns, time-step major.
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

      wts <- l$get_weights()
      dw_kern <- wts[[1L]] # (kW, C_in, depth_mult)
      k_w <- dim(dw_kern)[1L]
      c_in <- dim(dw_kern)[2L]
      depth_mult <- dim(dw_kern)[3L]
      c_out <- c_in * depth_mult
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

      dw_nms <- character(0L)
      dw_exprs <- character(0L)
      for (p in seq_len(w_out)) {
        p0 <- p - 1L
        for (c in seq_len(c_in)) {
          for (d in seq_len(depth_mult)) {
            terms <- character(0L)
            for (k in seq_len(k_w)) {
              k0 <- k - 1L
              w_pos <- p0 * stride + k0 * dilation - pad_l
              if (w_pos >= 0L && w_pos < w_in) {
                feat_nm <- in_exprs[w_pos * c_in + c]
                wt_val <- format_numeric(dw_kern[k, c, d])
                terms <- c(
                  terms,
                  paste0("(", backtick(feat_nm), " * ", wt_val, ")")
                )
              }
            }
            ch_out <- (c - 1L) * depth_mult + d
            b_str <- format_numeric(bias_v[ch_out])
            expr_str <- if (length(terms) == 0L) {
              b_str
            } else {
              paste0("(", paste(c(terms, b_str), collapse = " + "), ")")
            }
            nm <- paste0("orbital_dw_", lname, "_p", p, "_c", ch_out)
            dw_nms <- c(dw_nms, nm)
            dw_exprs <- c(dw_exprs, expr_str)
          }
        }
      }
      all_exprs[[lname]] <- stats::setNames(dw_exprs, dw_nms)
      assign(lname, dw_nms, envir = expr_reg)
    } else if (grepl("separableconv1d", cls)) {
      # SeparableConv1D ─ depthwise + pointwise 1-D convolution.
      # Keras weight layout:
      #   depthwise_kernel  : (kernel_size, in_channels, depth_multiplier)
      #   pointwise_kernel  : (1, in_channels * depth_multiplier, out_channels)
      #   bias              : (out_channels,)  [optional]
      # Two-stage process:
      #   1. Depthwise conv: C_in * depth_mult intermediate channels per timestep.
      #   2. Pointwise conv: 1x1 linear projection to C_out.
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

      wts <- l$get_weights()
      dw_kern <- wts[[1L]] # (kW, C_in, depth_mult)
      pw_kern <- wts[[2L]] # (1, C_in * depth_mult, C_out)
      k_w <- dim(dw_kern)[1L]
      c_in <- dim(dw_kern)[2L]
      depth_mult <- dim(dw_kern)[3L]
      c_mid <- c_in * depth_mult
      c_out <- dim(pw_kern)[3L]
      bias_v <- if (length(wts) >= 3L) as.numeric(wts[[3L]]) else numeric(c_out)

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

      # Stage 1: depthwise intermediates (no cross-channel mixing).
      dw_nms <- character(0L)
      dw_exprs <- character(0L)
      for (p in seq_len(w_out)) {
        p0 <- p - 1L
        for (c in seq_len(c_in)) {
          for (d in seq_len(depth_mult)) {
            terms <- character(0L)
            for (k in seq_len(k_w)) {
              k0 <- k - 1L
              w_pos <- p0 * stride + k0 * dilation - pad_l
              if (w_pos >= 0L && w_pos < w_in) {
                feat_nm <- in_exprs[w_pos * c_in + c]
                wt_val <- format_numeric(dw_kern[k, c, d])
                terms <- c(
                  terms,
                  paste0("(", backtick(feat_nm), " * ", wt_val, ")")
                )
              }
            }
            c2 <- (c - 1L) * depth_mult + d
            expr_str <- if (length(terms) == 0L) {
              "0"
            } else {
              paste0("(", paste(terms, collapse = " + "), ")")
            }
            nm <- paste0("orbital_sep_dw_", lname, "_p", p, "_c", c2)
            dw_nms <- c(dw_nms, nm)
            dw_exprs <- c(dw_exprs, expr_str)
          }
        }
      }

      # Stage 2: pointwise 1x1 projection to C_out.
      sep_nms <- character(0L)
      sep_exprs <- character(0L)
      for (p in seq_len(w_out)) {
        for (f in seq_len(c_out)) {
          terms <- character(0L)
          for (c2 in seq_len(c_mid)) {
            dw_nm <- paste0("orbital_sep_dw_", lname, "_p", p, "_c", c2)
            wt_val <- format_numeric(pw_kern[1L, c2, f])
            terms <- c(
              terms,
              paste0("(", backtick(dw_nm), " * ", wt_val, ")")
            )
          }
          b_str <- format_numeric(bias_v[f])
          expr_str <- if (length(terms) == 0L) {
            b_str
          } else {
            paste0("(", paste(c(terms, b_str), collapse = " + "), ")")
          }
          nm <- paste0("orbital_sep_", lname, "_p", p, "_f", f)
          sep_nms <- c(sep_nms, nm)
          sep_exprs <- c(sep_exprs, expr_str)
        }
      }
      all_exprs[[lname]] <- c(
        stats::setNames(dw_exprs, dw_nms),
        stats::setNames(sep_exprs, sep_nms)
      )
      assign(lname, sep_nms, envir = expr_reg)
    } else if (grepl("convlstm1d", cls)) {
      # ConvLSTM1D ─ unrolled for fixed-length sequences with spatial dim = 1.
      # Keras weight layout (IFCO gate order):
      #   kernel           : (kernel_size, in_channels, 4 * filters)
      #   recurrent_kernel : (kernel_size, filters,     4 * filters)
      #   bias             : (4 * filters,)  [optional]
      # Restriction: only spatial dimension S = 1, strides = 1, padding = "same".
      # With S=1 and padding="same", each spatial convolution reduces to a
      # single-position dot product using the centre kernel row (index pad_l).
      # This makes ConvLSTM1D equivalent to a plain LSTM after weight slicing.
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      wts <- l$get_weights() # kernel, recurrent_kernel [, bias]
      kernel <- wts[[1L]] # (k, C, 4F)
      rkernel <- wts[[2L]] # (k, F, 4F)
      k_size <- dim(kernel)[1L]
      C_in <- dim(kernel)[2L]
      F_filt <- as.integer(dim(kernel)[3L] / 4L)
      bias_v <- if (length(wts) >= 3L) as.numeric(wts[[3L]]) else numeric(4L * F_filt)

      cfg_l <- tryCatch(l$get_config(), error = function(e) list())
      pad_type <- tryCatch(
        tolower(as.character(cfg_l$padding)),
        error = function(e) "same"
      )
      strides <- tryCatch(as.integer(cfg_l$strides[[1L]]), error = function(e) 1L)
      if (is.na(strides)) strides <- 1L
      if (!nzchar(pad_type %||% "")) pad_type <- "same"
      if (pad_type != "same" || strides != 1L) {
        cli::cli_abort(c(
          "ConvLSTM1D layer {.val {lname}}: orbital only supports padding='same' and strides=1.",
          "i" = "Got padding={.val {pad_type}}, strides={.val {strides}}."
        ))
      }
      if (
        isTRUE(tryCatch(as.logical(cfg_l$stateful), error = function(e) FALSE))
      ) {
        cli::cli_abort(c(
          "ConvLSTM1D layer {.val {lname}}: stateful = TRUE is not supported by orbital.",
          "i" = "Only stateless ConvLSTM1Ds can be unrolled."
        ))
      }

      # Infer spatial dimension S from input_shape.
      # Keras input_shape for ConvLSTM1D: (batch, T, S, C_in).
      in_shape <- tryCatch(
        as.integer(unlist(l$input_shape)),
        error = function(e) NULL
      )
      in_shape_valid <- if (!is.null(in_shape)) in_shape[!is.na(in_shape)] else integer(0L)
      # in_shape_valid (after dropping NA batch dim): [T, S, C_in]
      if (length(in_shape_valid) >= 2L) {
        S_spatial <- as.integer(in_shape_valid[length(in_shape_valid) - 1L])
      } else {
        S_spatial <- NA_integer_
      }
      if (is.na(S_spatial) || S_spatial != 1L) {
        cli::cli_abort(c(
          "ConvLSTM1D layer {.val {lname}}: orbital only supports spatial dimension S = 1.",
          "i" = "Got S = {.val {S_spatial}}."
        ))
      }
      # With S=1: in_exprs has T_len * C_in features (the single spatial position is trivial).
      T_len <- as.integer(length(in_exprs) / C_in)

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

      # Centre kernel row (0-indexed pad_l → 1-indexed pad_l + 1L in R).
      pad_l <- as.integer((k_size - 1L) %/% 2L)
      # eff_kern  : (C_in, 4*F_filt) — equivalent to LSTM kernel  (I, 4H)
      # eff_rkernel: (F_filt, 4*F_filt) — equivalent to LSTM recurrent_kernel (H, 4H)
      eff_kern <- kernel[pad_l + 1L, , ]
      dim(eff_kern) <- c(C_in, 4L * F_filt)
      eff_rkernel <- rkernel[pad_l + 1L, , ]
      dim(eff_rkernel) <- c(F_filt, 4L * F_filt)

      # IFCO gate offsets (0-indexed): I=0, F_gate=F_filt, C_tilde=2F_filt, O=3F_filt
      g_offs <- c(0L, F_filt, 2L * F_filt, 3L * F_filt)
      W_gates <- lapply(seq_along(g_offs), function(g) {
        t(eff_kern[, (g_offs[g] + 1L):(g_offs[g] + F_filt)])
      })
      R_gates <- lapply(seq_along(g_offs), function(g) {
        t(eff_rkernel[, (g_offs[g] + 1L):(g_offs[g] + F_filt)])
      })
      B_gates <- lapply(seq_along(g_offs), function(g) {
        as.numeric(bias_v[(g_offs[g] + 1L):(g_offs[g] + F_filt)])
      })

      all_clstm_nms <- character(0L)
      all_clstm_exprs <- character(0L)
      all_H_nms <- list()

      H_prev_nms <- NULL # NULL = zero initial hidden state
      C_prev_nms <- NULL # NULL = zero initial cell state

      for (t in seq_len(T_len)) {
        x_t_exprs <- in_exprs[((t - 1L) * C_in + 1L):(t * C_in)]

        pre_gates <- lapply(seq_along(g_offs), function(g) {
          inp <- build_mlp_pre_act(W_gates[[g]], B_gates[[g]], x_t_exprs)
          if (is.null(H_prev_nms)) {
            inp
          } else {
            rec <- build_mlp_pre_act(R_gates[[g]], numeric(F_filt), H_prev_nms)
            paste0("(", inp, " + ", rec, ")")
          }
        })

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

        C_cur_nms <- paste0(
          "orbital_convlstm_", lname, "_C_t", t, "_h", seq_len(F_filt)
        )
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

        H_cur_nms <- paste0(
          "orbital_convlstm_", lname, "_H_t", t, "_h", seq_len(F_filt)
        )
        H_cur_exprs <- paste0("(", act_O, " * tanh(", backtick(C_cur_nms), "))")

        all_clstm_nms <- c(all_clstm_nms, C_cur_nms, H_cur_nms)
        all_clstm_exprs <- c(all_clstm_exprs, C_cur_exprs, H_cur_exprs)
        all_H_nms[[t]] <- H_cur_nms

        H_prev_nms <- H_cur_nms
        C_prev_nms <- C_cur_nms
      }

      all_exprs[[lname]] <- stats::setNames(all_clstm_exprs, all_clstm_nms)
      out_nms <- if (return_seq) {
        unlist(all_H_nms, use.names = FALSE)
      } else {
        H_prev_nms
      }
      assign(lname, out_nms, envir = expr_reg)
    } else if (
      grepl("conv1d", cls) && !grepl("depthwise|separable|transpose|2d|3d", cls)
    ) {
      # Conv1D ─ sliding-window 1-D convolution.
      # Keras weight layout:
      #   kernel  : (kernel_size, in_channels, filters)
      #   bias    : (filters,)  [optional]
      # orbital input convention : T_in × C_in flat columns, time-step major.
      #   in_exprs[(t-1)*C_in + c] = orbital feature for timestep t, channel c (1-indexed).
      #   This matches Keras 3's (batch, T_in, C_in) row-major flattening.
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
                feat_nm <- in_exprs[w_pos * c_in + c]
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
      # y_i = x_i / sqrt(max(x_1^2 + ... + x_n^2, 1e-7))
      # Matches Keras 3: keras.backend.epsilon() = 1e-7 (float32 default).
      # Only axis = -1 (normalise over feature dimension) is supported.
      unit_axis_raw <- tryCatch(
        as.integer(unlist(l$get_config()$axis)[[1L]]),
        error = function(e) -1L
      )
      if (is.na(unit_axis_raw)) {
        unit_axis_raw <- -1L
      }
      if (!unit_axis_raw %in% c(-1L, 1L)) {
        cli::cli_abort(
          c(
            "UnitNormalization layer {.val {lname}}: axis = {.val {unit_axis_raw}} is not supported.",
            "i" = "Only axis = -1 (feature-wise L2 normalisation) is supported by orbital."
          )
        )
      }
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      n_feat <- length(in_exprs)
      expr_bt <- backtick(in_exprs)
      norm_nm <- paste0("orbital_unitnorm_", lname, "_norm")
      sq_sum <- paste(paste0(expr_bt, "^2"), collapse = " + ")
      norm_expr <- paste0("sqrt(do.call(pmax, list(", sq_sum, ", 1e-7)))")
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
    } else if (grepl("permute", cls) && !grepl("2d|3d", cls)) {
      # Permute: reorder the axes (dimensions) of the input tensor.
      # In the tabular/1D context the input is [T × C] (time-step major flat).
      # cfg$dims is 1-indexed (Keras convention) over the non-batch axes.
      # For a 2-D input the only supported permutations are (1,2) (no-op) and (2,1).
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      in_names <- in_exprs # column-name vector

      cfg <- tryCatch(l$get_config(), error = function(e) list())
      dims <- tryCatch(as.integer(unlist(cfg$dims)), error = function(e) {
        c(1L, 2L)
      })

      if (length(dims) == 2L && all(dims == c(1L, 2L))) {
        # No-op permutation (1,2): pass through unchanged.
        perm_names <- in_names
      } else if (length(dims) == 2L && all(dims == c(2L, 1L))) {
        # Transpose: swap T and C.
        in_shape <- tryCatch(
          as.integer(unlist(l$input_shape)),
          error = function(e) NULL
        )
        C_feat <- if (!is.null(in_shape) && length(in_shape) >= 1L) {
          tail(in_shape[!is.na(in_shape)], 1L)
        } else {
          1L
        }
        T_in <- length(in_names) / C_feat
        # Transposed: iterate over C first then T (column-major → row-major swap)
        perm_names <- character(length(in_names))
        for (c_i in seq_len(C_feat)) {
          for (t_i in seq_len(T_in)) {
            perm_names[[(c_i - 1L) * T_in + t_i]] <- in_names[[
              (t_i - 1L) * C_feat + c_i
            ]]
          }
        }
      } else {
        cli::cli_abort(
          "Keras Permute layer {.val {lname}}: unsupported dims {paste(dims, collapse=',')}. Only (1,2) and (2,1) are supported."
        )
      }
      perm_out_nms <- paste0(
        "orbital_permute_",
        lname,
        "_h",
        seq_along(perm_names)
      )
      all_exprs[[lname]] <- stats::setNames(perm_names, perm_out_nms)
      assign(lname, perm_out_nms, envir = expr_reg)
    } else if (grepl("cropping1d", cls)) {
      # Cropping1D: remove timesteps from the beginning and end of the sequence.
      # cfg$cropping = [left, right] (number of timesteps to remove from each end).
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

      cfg <- tryCatch(l$get_config(), error = function(e) list())
      cropping_raw <- tryCatch(
        as.integer(unlist(cfg$cropping)),
        error = function(e) c(1L, 1L)
      )
      crop_left <- if (length(cropping_raw) >= 1L) cropping_raw[[1L]] else 1L
      crop_right <- if (length(cropping_raw) >= 2L) {
        cropping_raw[[2L]]
      } else {
        cropping_raw[[1L]]
      }

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
      T_out <- T_in - crop_left - crop_right
      if (T_out <= 0L) {
        cli::cli_abort(
          "Keras Cropping1D layer {.val {lname}}: cropping ({crop_left},{crop_right}) removes all {T_in} timesteps."
        )
      }

      start_idx <- crop_left * C_feat + 1L
      end_idx <- (T_in - crop_right) * C_feat
      out_exprs <- in_exprs[start_idx:end_idx]
      out_nms <- paste0("orbital_crop_", lname, "_h", seq_along(out_exprs))
      all_exprs[[lname]] <- stats::setNames(out_exprs, out_nms)
      assign(lname, out_nms, envir = expr_reg)
    } else if (grepl("repeatvector", cls)) {
      # RepeatVector: replicate the (flat) input feature vector n times.
      # cfg$n = repetition count; output shape = [n × C_in].
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

      cfg <- tryCatch(l$get_config(), error = function(e) list())
      n_rep <- tryCatch(as.integer(cfg[["n"]]), error = function(e) 1L)
      if (is.na(n_rep) || n_rep < 1L) {
        cli::cli_abort(
          "Keras RepeatVector layer {.val {lname}}: n must be a positive integer, got {n_rep}."
        )
      }

      out_exprs <- rep(in_exprs, times = n_rep)
      out_nms <- paste0("orbital_repvec_", lname, "_h", seq_along(out_exprs))
      all_exprs[[lname]] <- stats::setNames(out_exprs, out_nms)
      assign(lname, out_nms, envir = expr_reg)
    } else if (
      grepl("\\battention\\b", cls, perl = TRUE) && !grepl("multihead", cls)
    ) {
      # Attention (Luong / dot-product): single-head dot-product soft-attention.
      # Inputs (inbound): [query, value] or [query, value, key].
      #   query shape: (T_q, D)  — flat layout T_q * D columns
      #   value shape: (T_v, D_v) — flat layout T_v * D_v columns
      #   key   shape: (T_k, D)  — defaults to value if not provided
      # use_scale = FALSE (default): no learned scale.
      # Algorithm:
      #   score[q, k] = sum_d query[q,d] * key[k,d]
      #   attn[q, k]  = softmax over k (max-stabilised)
      #   out[q, d_v] = sum_k attn[q,k] * value[k, d_v]
      inbound <- topo_map[[lname]]
      if (is.null(inbound) || length(inbound) < 2L) {
        cli::cli_abort(
          "Keras Attention layer {.val {lname}} requires at least 2 inbound inputs (query, value)."
        )
      }
      q_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      v_exprs <- get(inbound[2L], envir = expr_reg, inherits = FALSE)
      k_exprs <- if (length(inbound) >= 3L) {
        get(inbound[3L], envir = expr_reg, inherits = FALSE)
      } else {
        v_exprs # key defaults to value
      }

      cfg_att <- tryCatch(l$get_config(), error = function(e) list())
      use_scale_att <- tryCatch(
        as.logical(cfg_att$use_scale),
        error = function(e) FALSE
      )
      if (isTRUE(use_scale_att) && length(l$get_weights()) > 0L) {
        cli::cli_abort(
          "Keras Attention layer {.val {lname}}: use_scale=TRUE is not yet supported."
        )
      }

      # Infer shapes from input sizes.  Both query and key must have same depth D.
      # For self-attention query==key, D_q must equal D_k.
      # We infer T_q, D from q_exprs length, T_v and D_v from v_exprs,
      # T_k and D_k from k_exprs.  We need D_q == D_k for dot-product.
      # The simplest inference path: assume square sequence & same depth.
      in_shape_q <- tryCatch(
        as.integer(unlist(l$input_spec[[1L]]$shape)),
        error = function(e) NULL
      )
      in_shape_v <- tryCatch(
        as.integer(unlist(l$input_spec[[2L]]$shape)),
        error = function(e) NULL
      )

      D_q <- if (!is.null(in_shape_q) && length(in_shape_q) >= 1L) {
        tail(in_shape_q[!is.na(in_shape_q)], 1L)
      } else {
        # Fallback: assume square — length = T^2 or T * D; try sqrt
        as.integer(sqrt(length(q_exprs)))
      }
      T_q <- as.integer(length(q_exprs) / D_q)

      D_v <- if (!is.null(in_shape_v) && length(in_shape_v) >= 1L) {
        tail(in_shape_v[!is.na(in_shape_v)], 1L)
      } else {
        D_q
      }
      T_v <- as.integer(length(v_exprs) / D_v)
      T_k <- as.integer(length(k_exprs) / D_q)

      # Compute scores: score[q_i, k_j] = sum_d q_exprs[(q_i-1)*D_q + d] * k_exprs[(k_j-1)*D_q + d]
      # Then max-stabilised softmax over k for each q_i.
      att_out_exprs <- character(0L)
      att_out_nms <- character(0L)

      score_nms <- matrix(
        paste0(
          "orbital_att_",
          lname,
          "_sc_q",
          rep(seq_len(T_q), each = T_k),
          "_k",
          rep(seq_len(T_k), T_q)
        ),
        nrow = T_q,
        ncol = T_k
      )
      score_exprs <- matrix("", nrow = T_q, ncol = T_k)
      for (q_i in seq_len(T_q)) {
        for (k_j in seq_len(T_k)) {
          terms <- vapply(
            seq_len(D_q),
            function(d) {
              paste0(
                "(",
                backtick(q_exprs[(q_i - 1L) * D_q + d]),
                " * ",
                backtick(k_exprs[(k_j - 1L) * D_q + d]),
                ")"
              )
            },
            character(1L)
          )
          score_exprs[q_i, k_j] <- paste0(
            "(",
            paste(terms, collapse = " + "),
            ")"
          )
        }
      }

      # For each q_i, compute softmax over k dimension (max-stabilised).
      for (q_i in seq_len(T_q)) {
        sc_names <- score_nms[q_i, ]
        sc_expr_q <- score_exprs[q_i, ]

        # Register score intermediates
        for (k_j in seq_len(T_k)) {
          all_exprs[[lname]] <- c(
            all_exprs[[lname]],
            stats::setNames(sc_expr_q[k_j], sc_names[k_j])
          )
        }

        max_nm <- paste0("orbital_att_", lname, "_max_q", q_i)
        max_expr <- paste0(
          "pmax(",
          paste(backtick(sc_names), collapse = ", "),
          ")"
        )
        all_exprs[[lname]] <- c(
          all_exprs[[lname]],
          stats::setNames(max_expr, max_nm)
        )

        exp_nms <- paste0(
          "orbital_att_",
          lname,
          "_exp_q",
          q_i,
          "_k",
          seq_len(T_k)
        )
        exp_exprs <- vapply(
          sc_names,
          function(s) paste0("exp(", backtick(s), " - ", backtick(max_nm), ")"),
          character(1L)
        )
        for (k_j in seq_len(T_k)) {
          all_exprs[[lname]] <- c(
            all_exprs[[lname]],
            stats::setNames(exp_exprs[k_j], exp_nms[k_j])
          )
        }

        sum_nm <- paste0("orbital_att_", lname, "_sum_q", q_i)
        sum_expr <- paste0("(", paste(backtick(exp_nms), collapse = " + "), ")")
        all_exprs[[lname]] <- c(
          all_exprs[[lname]],
          stats::setNames(sum_expr, sum_nm)
        )

        attn_nms <- paste0(
          "orbital_att_",
          lname,
          "_attn_q",
          q_i,
          "_k",
          seq_len(T_k)
        )
        attn_exprs <- vapply(
          exp_nms,
          function(e) paste0("(", backtick(e), " / ", backtick(sum_nm), ")"),
          character(1L)
        )
        for (k_j in seq_len(T_k)) {
          all_exprs[[lname]] <- c(
            all_exprs[[lname]],
            stats::setNames(attn_exprs[k_j], attn_nms[k_j])
          )
        }

        # Output: weighted sum over value
        for (d_v in seq_len(D_v)) {
          terms <- vapply(
            seq_len(T_v),
            function(k_j) {
              paste0(
                "(",
                backtick(attn_nms[k_j]),
                " * ",
                backtick(v_exprs[(k_j - 1L) * D_v + d_v]),
                ")"
              )
            },
            character(1L)
          )
          out_expr <- paste0("(", paste(terms, collapse = " + "), ")")
          out_nm <- paste0("orbital_att_", lname, "_out_q", q_i, "_d", d_v)
          att_out_exprs <- c(att_out_exprs, out_expr)
          att_out_nms <- c(att_out_nms, out_nm)
          all_exprs[[lname]] <- c(
            all_exprs[[lname]],
            stats::setNames(out_expr, out_nm)
          )
        }
      }
      assign(lname, att_out_nms, envir = expr_reg)
    } else if (grepl("additiveattention", cls)) {
      # AdditiveAttention (Bahdanau attention):
      # score[q,k] = scale * sum_d tanh(query[q,d] + key[k,d])
      # where scale is 1.0 if use_scale=FALSE (the Keras default).
      # Softmax and output weighted-sum are identical to Attention.
      inbound <- topo_map[[lname]]
      if (is.null(inbound) || length(inbound) < 2L) {
        cli::cli_abort(
          "Keras AdditiveAttention layer {.val {lname}} requires at least 2 inbound inputs (query, value)."
        )
      }
      q_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      v_exprs <- get(inbound[2L], envir = expr_reg, inherits = FALSE)
      k_exprs <- if (length(inbound) >= 3L) {
        get(inbound[3L], envir = expr_reg, inherits = FALSE)
      } else {
        v_exprs
      }

      cfg_aa <- tryCatch(l$get_config(), error = function(e) list())
      use_scale_aa <- tryCatch(
        as.logical(cfg_aa$use_scale),
        error = function(e) FALSE
      )
      aa_scale <- if (isTRUE(use_scale_aa)) {
        wts_aa <- tryCatch(l$get_weights(), error = function(e) list())
        if (length(wts_aa) >= 1L) as.numeric(wts_aa[[1L]])[1L] else 1.0
      } else {
        1.0
      }

      in_shape_q_aa <- tryCatch(
        as.integer(unlist(l$input_spec[[1L]]$shape)),
        error = function(e) NULL
      )
      in_shape_v_aa <- tryCatch(
        as.integer(unlist(l$input_spec[[2L]]$shape)),
        error = function(e) NULL
      )
      D_q_aa <- if (!is.null(in_shape_q_aa) && length(in_shape_q_aa) >= 1L) {
        tail(in_shape_q_aa[!is.na(in_shape_q_aa)], 1L)
      } else {
        as.integer(sqrt(length(q_exprs)))
      }
      T_q_aa <- as.integer(length(q_exprs) / D_q_aa)
      D_v_aa <- if (!is.null(in_shape_v_aa) && length(in_shape_v_aa) >= 1L) {
        tail(in_shape_v_aa[!is.na(in_shape_v_aa)], 1L)
      } else {
        D_q_aa
      }
      T_v_aa <- as.integer(length(v_exprs) / D_v_aa)
      T_k_aa <- as.integer(length(k_exprs) / D_q_aa)

      att_out_exprs_aa <- character(0L)
      att_out_nms_aa <- character(0L)

      # Scores: sum_d tanh(q[d] + k[d]) * scale
      score_nms_aa <- matrix(
        paste0(
          "orbital_addatt_",
          lname,
          "_sc_q",
          rep(seq_len(T_q_aa), each = T_k_aa),
          "_k",
          rep(seq_len(T_k_aa), T_q_aa)
        ),
        nrow = T_q_aa,
        ncol = T_k_aa
      )
      score_exprs_aa <- matrix("", nrow = T_q_aa, ncol = T_k_aa)
      for (q_i in seq_len(T_q_aa)) {
        for (k_j in seq_len(T_k_aa)) {
          terms <- vapply(
            seq_len(D_q_aa),
            function(d) {
              paste0(
                "tanh(",
                backtick(q_exprs[(q_i - 1L) * D_q_aa + d]),
                " + ",
                backtick(k_exprs[(k_j - 1L) * D_q_aa + d]),
                ")"
              )
            },
            character(1L)
          )
          score_exprs_aa[q_i, k_j] <- paste0(
            "(",
            aa_scale,
            " * (",
            paste(terms, collapse = " + "),
            "))"
          )
        }
      }

      # Softmax + weighted output (identical structure to Attention)
      for (q_i in seq_len(T_q_aa)) {
        sc_names_aa <- score_nms_aa[q_i, ]
        sc_expr_q_aa <- score_exprs_aa[q_i, ]
        for (k_j in seq_len(T_k_aa)) {
          all_exprs[[lname]] <- c(
            all_exprs[[lname]],
            stats::setNames(sc_expr_q_aa[k_j], sc_names_aa[k_j])
          )
        }
        max_nm_aa <- paste0("orbital_addatt_", lname, "_max_q", q_i)
        max_expr_aa <- paste0(
          "pmax(",
          paste(backtick(sc_names_aa), collapse = ", "),
          ")"
        )
        all_exprs[[lname]] <- c(
          all_exprs[[lname]],
          stats::setNames(max_expr_aa, max_nm_aa)
        )
        exp_nms_aa <- paste0(
          "orbital_addatt_",
          lname,
          "_exp_q",
          q_i,
          "_k",
          seq_len(T_k_aa)
        )
        exp_exprs_aa <- vapply(
          sc_names_aa,
          function(s) {
            paste0("exp(", backtick(s), " - ", backtick(max_nm_aa), ")")
          },
          character(1L)
        )
        for (k_j in seq_len(T_k_aa)) {
          all_exprs[[lname]] <- c(
            all_exprs[[lname]],
            stats::setNames(exp_exprs_aa[k_j], exp_nms_aa[k_j])
          )
        }
        sum_nm_aa <- paste0("orbital_addatt_", lname, "_sum_q", q_i)
        sum_expr_aa <- paste0(
          "(",
          paste(backtick(exp_nms_aa), collapse = " + "),
          ")"
        )
        all_exprs[[lname]] <- c(
          all_exprs[[lname]],
          stats::setNames(sum_expr_aa, sum_nm_aa)
        )
        attn_nms_aa <- paste0(
          "orbital_addatt_",
          lname,
          "_attn_q",
          q_i,
          "_k",
          seq_len(T_k_aa)
        )
        attn_exprs_aa <- vapply(
          exp_nms_aa,
          function(e) paste0("(", backtick(e), " / ", backtick(sum_nm_aa), ")"),
          character(1L)
        )
        for (k_j in seq_len(T_k_aa)) {
          all_exprs[[lname]] <- c(
            all_exprs[[lname]],
            stats::setNames(attn_exprs_aa[k_j], attn_nms_aa[k_j])
          )
        }
        for (d_v in seq_len(D_v_aa)) {
          terms <- vapply(
            seq_len(T_v_aa),
            function(k_j) {
              paste0(
                "(",
                backtick(attn_nms_aa[k_j]),
                " * ",
                backtick(v_exprs[(k_j - 1L) * D_v_aa + d_v]),
                ")"
              )
            },
            character(1L)
          )
          out_expr_aa <- paste0("(", paste(terms, collapse = " + "), ")")
          out_nm_aa <- paste0(
            "orbital_addatt_",
            lname,
            "_out_q",
            q_i,
            "_d",
            d_v
          )
          att_out_exprs_aa <- c(att_out_exprs_aa, out_expr_aa)
          att_out_nms_aa <- c(att_out_nms_aa, out_nm_aa)
          all_exprs[[lname]] <- c(
            all_exprs[[lname]],
            stats::setNames(out_expr_aa, out_nm_aa)
          )
        }
      }
      assign(lname, att_out_nms_aa, envir = expr_reg)
    } else if (grepl("timedistributed", cls)) {
      # TimeDistributed: apply a layer independently to each timestep.
      # Supports Dense inner layer; other inner types raise cli_abort.
      # Input:  T_in x C_in flat columns (time-step major).
      # Output: T_in x units flat columns (time-step major).
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

      inner <- tryCatch(l$layer, error = function(e) NULL)
      if (is.null(inner)) {
        cli::cli_abort(
          "Keras TimeDistributed layer {.val {lname}}: cannot access wrapped layer."
        )
      }
      inner_cls <- tolower(class(inner)[1L])

      if (grepl("dense", inner_cls)) {
        inner_wts <- inner$get_weights()
        if (length(inner_wts) < 1L) {
          cli::cli_abort(
            "Keras TimeDistributed(Dense) layer {.val {lname}}: no weights found."
          )
        }
        kern_td <- t(inner_wts[[1L]]) # (units x C_in)
        bias_td <- if (length(inner_wts) >= 2L) {
          as.numeric(inner_wts[[2L]])
        } else {
          numeric(nrow(kern_td))
        }
        units <- nrow(kern_td)
        c_in_td <- ncol(kern_td)
        T_steps <- as.integer(length(in_exprs) / c_in_td)

        activation_config_td <- tryCatch(
          inner$get_config()$activation,
          error = function(e) NULL
        )
        activation_td <- if (is.null(activation_config_td)) {
          "linear"
        } else if (is.list(activation_config_td)) {
          tolower(as.character(activation_config_td$class_name)[1L])
        } else {
          tolower(as.character(activation_config_td)[1L])
        }
        if (!nzchar(activation_td)) {
          activation_td <- "linear"
        }
        act_alpha_td <- if (
          is.list(activation_config_td) &&
            !is.null(activation_config_td[["config"]])
        ) {
          activation_config_td[["config"]][["alpha"]]
        } else {
          NULL
        }

        td_nms <- character(0L)
        td_exprs <- character(0L)
        for (t in seq_len(T_steps)) {
          t_in <- in_exprs[((t - 1L) * c_in_td + 1L):(t * c_in_td)]
          pre_act <- build_mlp_pre_act(kern_td, bias_td, t_in)
          act_str <- vapply(
            pre_act,
            function(z) activation_expr(activation_td, z, alpha = act_alpha_td),
            character(1L)
          )
          step_nms <- paste0(
            "orbital_td_",
            lname,
            "_t",
            t,
            "_h",
            seq_len(units)
          )
          td_nms <- c(td_nms, step_nms)
          td_exprs <- c(td_exprs, act_str)
        }
        all_exprs[[lname]] <- stats::setNames(td_exprs, td_nms)
        assign(lname, td_nms, envir = expr_reg)
      } else {
        cli::cli_abort(c(
          "Keras TimeDistributed layer {.val {lname}} wraps unsupported inner layer type {.cls {inner_cls}}.",
          "i" = "orbital currently supports TimeDistributed(Dense) only."
        ))
      }
    } else if (grepl("\\bembedding\\b", cls, perl = TRUE)) {
      # Embedding: integer index lookup into a dense weight matrix.
      # Keras weight layout:
      #   embeddings : (vocab_size, embed_dim)
      # Input:  T flat integer columns (one token index per timestep, 0-indexed).
      # Output: T x embed_dim flat columns (time-step major).
      inbound <- topo_map[[lname]]
      in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)

      wts <- l$get_weights()
      emb_mat <- wts[[1L]] # (vocab_size, embed_dim)
      vocab_size <- dim(emb_mat)[1L]
      embed_dim <- dim(emb_mat)[2L]

      if (vocab_size > 50000L) {
        cli::cli_abort(c(
          "Keras Embedding layer {.val {lname}}: vocabulary size {vocab_size} exceeds the",
          " maximum supported limit of 50,000.",
          "x" = "Expanding this layer would generate >50,000 CASE WHEN branches per output column,",
          "     which most SQL engines cannot compile or execute.",
          "i" = "Consider replacing the Embedding layer with a pre-computed lookup table and",
          "     joining the index column against it inside the database instead."
        ))
      }

      if (vocab_size > 10000L) {
        cli::cli_warn(c(
          "Keras Embedding layer {.val {lname}}: vocabulary size {vocab_size} is large.",
          "i" = paste(
            "Generated case_when expressions may be very long.",
            "Consider reducing vocabulary size."
          )
        ))
      }

      T_in <- length(in_exprs) # one column per token position
      emb_nms <- character(0L)
      emb_exprs <- character(0L)
      for (t in seq_len(T_in)) {
        in_col <- in_exprs[t]
        for (d in seq_len(embed_dim)) {
          cases <- vapply(
            seq_len(vocab_size),
            function(i) {
              paste0(
                backtick(in_col),
                " == ",
                i - 1L,
                "L ~ ",
                format_numeric(emb_mat[i, d])
              )
            },
            character(1L)
          )
          expr_str <- paste0(
            "dplyr::case_when(",
            paste(cases, collapse = ", "),
            ", TRUE ~ NA_real_)"
          )
          nm <- paste0("orbital_emb_", lname, "_t", t, "_d", d)
          emb_nms <- c(emb_nms, nm)
          emb_exprs <- c(emb_exprs, expr_str)
        }
      }
      all_exprs[[lname]] <- stats::setNames(emb_exprs, emb_nms)
      assign(lname, emb_nms, envir = expr_reg)
    } else if (grepl("multiheadattention", cls)) {
      # MultiHeadAttention: scaled dot-product self-attention or cross-attention.
      # Supports: fixed-length sequences; causal_mask = FALSE; use_bias = TRUE/FALSE.
      # Input layout: time-step major (T × C_in) flat vector.
      # Weight extraction via reticulate::py_get_attr on internal EinsumDense sub-layers.
      inbound <- topo_map[[lname]]
      q_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
      kv_exprs <- get(
        inbound[min(2L, length(inbound))],
        envir = expr_reg,
        inherits = FALSE
      )

      cfg <- tryCatch(l$get_config(), error = function(e) list())
      num_heads <- as.integer(cfg$num_heads %||% 1L)
      key_dim <- as.integer(cfg$key_dim %||% 1L)
      value_dim <- as.integer(cfg$value_dim %||% key_dim)
      use_bias <- isTRUE(as.logical(cfg$use_bias %||% TRUE))

      in_shape <- tryCatch(
        as.integer(unlist(l$input_shape[[1L]])),
        error = function(e) NULL
      )
      C_in <- if (!is.null(in_shape)) {
        tail(in_shape[!is.na(in_shape)], 1L)
      } else {
        1L
      }
      T_q <- as.integer(length(q_exprs) / C_in)
      T_kv <- as.integer(length(kv_exprs) / C_in)

      out_shape <- tryCatch(
        as.integer(unlist(l$output_shape)),
        error = function(e) NULL
      )
      C_out <- if (!is.null(out_shape)) {
        tail(out_shape[!is.na(out_shape)], 1L)
      } else {
        C_in
      }

      # Retrieve kernel and bias from an EinsumDense sub-layer (handles _attr naming).
      .mha_wts <- function(attr_name) {
        sub <- tryCatch(
          reticulate::py_get_attr(l, attr_name),
          error = function(e) NULL
        )
        if (is.null(sub)) {
          return(NULL)
        }
        k <- tryCatch(
          as.array(reticulate::py_to_r(sub$kernel)),
          error = function(e) NULL
        )
        b <- if (use_bias) {
          tryCatch(
            as.array(reticulate::py_to_r(sub$bias)),
            error = function(e) NULL
          )
        } else {
          NULL
        }
        if (is.null(k)) {
          return(NULL)
        }
        list(kernel = k, bias = b)
      }

      wq <- .mha_wts("_query_dense")
      wk <- .mha_wts("_key_dense")
      wv <- .mha_wts("_value_dense")
      wo <- .mha_wts("_output_dense")

      if (is.null(wq) || is.null(wk) || is.null(wv) || is.null(wo)) {
        cli::cli_abort(
          "MultiHeadAttention {.val {lname}}: could not extract sub-layer projection weights.",
          "i" = "Ensure keras3 >= 3.0 with reticulate access to _query_dense, _key_dense, etc."
        )
      }

      # Determine weight accessor functions given that Keras3 EinsumDense stores
      # Q/K kernels as 3-D arrays.  Two known layouts:
      #   (a) (C_in, key_dim, num_heads)  — einsum "abc,ced->abde" with d=num_heads,e=key_dim
      #   (b) (C_in, num_heads, key_dim)  — alternative layout
      .make_proj_at <- function(k, dim2_size, dim3_size, name) {
        d <- dim(k)
        if (length(d) != 3L) {
          cli::cli_abort(
            "MHA {.val {lname}}: {name} kernel must be 3-D, got {length(d)}D."
          )
        }
        if (d[1L] != C_in) {
          cli::cli_abort(
            "MHA {.val {lname}}: {name} kernel dim1={d[1L]} does not match C_in={C_in}."
          )
        }
        if (d[2L] == dim2_size && d[3L] == dim3_size) {
          function(c_idx, idx2, idx3) k[c_idx, idx2, idx3]
        } else if (d[2L] == dim3_size && d[3L] == dim2_size) {
          function(c_idx, idx2, idx3) k[c_idx, idx3, idx2]
        } else {
          cli::cli_abort(
            "MHA {.val {lname}}: {name} kernel dims ({d}) cannot be reconciled with ",
            "expected ({C_in}, {dim2_size}, {dim3_size}) or transposed."
          )
        }
      }

      # Q/K: (C_in, ?, ?) with dims key_dim and num_heads in some order.
      # After .make_proj_at, call as wq_at(c, key_dim_idx, head_idx) → scalar.
      wq_at <- .make_proj_at(wq$kernel, key_dim, num_heads, "Q")
      wk_at <- .make_proj_at(wk$kernel, key_dim, num_heads, "K")
      # V: (C_in, ?, ?) with dims value_dim and num_heads.
      wv_at <- .make_proj_at(wv$kernel, value_dim, num_heads, "V")

      # Q bias accessor: shape (key_dim, num_heads) or (num_heads, key_dim).
      .make_bias_at2 <- function(b, d2, d3, name) {
        if (is.null(b)) {
          return(function(i2, i3) "0")
        }
        di <- dim(b)
        if (length(di) == 2L && di[1L] == d2 && di[2L] == d3) {
          function(i2, i3) format_numeric(b[i2, i3])
        } else if (length(di) == 2L && di[1L] == d3 && di[2L] == d2) {
          function(i2, i3) format_numeric(b[i3, i2])
        } else {
          function(i2, i3) "0"
        }
      }
      bq_at <- .make_bias_at2(wq$bias, key_dim, num_heads, "Q_bias")
      bk_at <- .make_bias_at2(wk$bias, key_dim, num_heads, "K_bias")
      bv_at <- .make_bias_at2(wv$bias, value_dim, num_heads, "V_bias")

      # Output projection: (num_heads, value_dim, C_out) or transposed.
      wo_d <- dim(wo$kernel)
      if (
        length(wo_d) == 3L && wo_d[1L] == num_heads && wo_d[2L] == value_dim
      ) {
        wo_at <- function(h, dv, do_) wo$kernel[h, dv, do_]
        C_out_w <- wo_d[3L]
      } else if (
        length(wo_d) == 3L && wo_d[1L] == C_out && wo_d[2L] == num_heads
      ) {
        # (C_out, num_heads, value_dim) layout
        wo_at <- function(h, dv, do_) wo$kernel[do_, h, dv]
        C_out_w <- wo_d[1L]
      } else {
        cli::cli_abort(
          "MHA {.val {lname}}: output kernel dims {wo_d} unrecognised."
        )
      }
      bo_at <- function(do_) {
        if (!is.null(wo$bias) && length(wo$bias) >= do_) {
          format_numeric(wo$bias[do_])
        } else {
          "0"
        }
      }

      # Helper: column name for an input at (timestep t, channel c)
      q_at <- function(t, c) backtick(q_exprs[(t - 1L) * C_in + c])
      kv_at <- function(t, c) backtick(kv_exprs[(t - 1L) * C_in + c])

      mha_nms <- character(0)
      mha_exprs <- character(0)

      # Q[q, h, d_k] = sum_c input_q[q, c] * W_q(c, d_k, h) + b_q(d_k, h)
      q_nm <- array(NA_character_, dim = c(T_q, num_heads, key_dim))
      for (q in seq_len(T_q)) {
        for (h in seq_len(num_heads)) {
          for (d in seq_len(key_dim)) {
            terms <- vapply(
              seq_len(C_in),
              function(c) {
                paste0(q_at(q, c), " * ", format_numeric(wq_at(c, d, h)))
              },
              character(1L)
            )
            nm <- paste0("orbital_mha_", lname, "_Q_q", q, "_h", h, "_d", d)
            mha_nms <- c(mha_nms, nm)
            mha_exprs <- c(
              mha_exprs,
              paste0(
                "(",
                paste(terms, collapse = " + "),
                " + ",
                bq_at(d, h),
                ")"
              )
            )
            q_nm[q, h, d] <- nm
          }
        }
      }

      # K[k, h, d_k] = sum_c input_kv[k, c] * W_k(c, d_k, h) + b_k(d_k, h)
      k_nm <- array(NA_character_, dim = c(T_kv, num_heads, key_dim))
      for (k in seq_len(T_kv)) {
        for (h in seq_len(num_heads)) {
          for (d in seq_len(key_dim)) {
            terms <- vapply(
              seq_len(C_in),
              function(c) {
                paste0(kv_at(k, c), " * ", format_numeric(wk_at(c, d, h)))
              },
              character(1L)
            )
            nm <- paste0("orbital_mha_", lname, "_K_k", k, "_h", h, "_d", d)
            mha_nms <- c(mha_nms, nm)
            mha_exprs <- c(
              mha_exprs,
              paste0(
                "(",
                paste(terms, collapse = " + "),
                " + ",
                bk_at(d, h),
                ")"
              )
            )
            k_nm[k, h, d] <- nm
          }
        }
      }

      # V[k, h, d_v] = sum_c input_kv[k, c] * W_v(c, d_v, h) + b_v(d_v, h)
      v_nm <- array(NA_character_, dim = c(T_kv, num_heads, value_dim))
      for (k in seq_len(T_kv)) {
        for (h in seq_len(num_heads)) {
          for (d_v in seq_len(value_dim)) {
            terms <- vapply(
              seq_len(C_in),
              function(c) {
                paste0(kv_at(k, c), " * ", format_numeric(wv_at(c, d_v, h)))
              },
              character(1L)
            )
            nm <- paste0("orbital_mha_", lname, "_V_k", k, "_h", h, "_dv", d_v)
            mha_nms <- c(mha_nms, nm)
            mha_exprs <- c(
              mha_exprs,
              paste0(
                "(",
                paste(terms, collapse = " + "),
                " + ",
                bv_at(d_v, h),
                ")"
              )
            )
            v_nm[k, h, d_v] <- nm
          }
        }
      }

      # Scaled attention scores and softmax (max-stabilised to prevent exp() overflow).
      # For each (q, h) pair: subtract max_{k'} score before exp(), consistent with
      # the standalone Softmax branches in this file.
      scale <- 1.0 / sqrt(as.numeric(key_dim))
      attn_nm <- array(NA_character_, dim = c(T_q, T_kv, num_heads))
      for (q in seq_len(T_q)) {
        for (h in seq_len(num_heads)) {
          # Collect raw scaled score expressions for all key positions
          score_exprs <- character(T_kv)
          for (k in seq_len(T_kv)) {
            dot_terms <- vapply(
              seq_len(key_dim),
              function(d) {
                paste0(backtick(q_nm[q, h, d]), " * ", backtick(k_nm[k, h, d]))
              },
              character(1L)
            )
            score_exprs[k] <- paste0(
              "((",
              paste(dot_terms, collapse = " + "),
              ") * ",
              format_numeric(scale),
              ")"
            )
          }

          # Max-stabilisation: compute row-wise max of scores for (q, h)
          max_nm <- paste0("orbital_mha_", lname, "_smax_q", q, "_h", h)
          mha_nms <- c(mha_nms, max_nm)
          mha_exprs <- c(
            mha_exprs,
            paste0(
              "do.call(pmax, list(",
              paste(score_exprs, collapse = ", "),
              "))"
            )
          )

          exp_nms <- character(T_kv)
          for (k in seq_len(T_kv)) {
            exp_nm <- paste0(
              "orbital_mha_",
              lname,
              "_exp_q",
              q,
              "_k",
              k,
              "_h",
              h
            )
            mha_nms <- c(mha_nms, exp_nm)
            mha_exprs <- c(
              mha_exprs,
              paste0("exp(", score_exprs[k], " - `", max_nm, "`)")
            )
            exp_nms[k] <- exp_nm
          }
          sumexp_nm <- paste0("orbital_mha_", lname, "_sumexp_q", q, "_h", h)
          mha_nms <- c(mha_nms, sumexp_nm)
          mha_exprs <- c(
            mha_exprs,
            paste0("(", paste(backtick(exp_nms), collapse = " + "), ")")
          )
          for (k in seq_len(T_kv)) {
            a_nm <- paste0(
              "orbital_mha_",
              lname,
              "_attn_q",
              q,
              "_k",
              k,
              "_h",
              h
            )
            mha_nms <- c(mha_nms, a_nm)
            mha_exprs <- c(
              mha_exprs,
              paste0("(", backtick(exp_nms[k]), " / ", backtick(sumexp_nm), ")")
            )
            attn_nm[q, k, h] <- a_nm
          }
        }
      }

      # Head outputs: head[q, h, d_v] = sum_k attn[q,k,h] * V[k,h,d_v]
      hout_nm <- array(NA_character_, dim = c(T_q, num_heads, value_dim))
      for (q in seq_len(T_q)) {
        for (h in seq_len(num_heads)) {
          for (d_v in seq_len(value_dim)) {
            terms <- vapply(
              seq_len(T_kv),
              function(k) {
                paste0(
                  backtick(attn_nm[q, k, h]),
                  " * ",
                  backtick(v_nm[k, h, d_v])
                )
              },
              character(1L)
            )
            nm <- paste0(
              "orbital_mha_",
              lname,
              "_hout_q",
              q,
              "_h",
              h,
              "_dv",
              d_v
            )
            mha_nms <- c(mha_nms, nm)
            mha_exprs <- c(
              mha_exprs,
              paste0("(", paste(terms, collapse = " + "), ")")
            )
            hout_nm[q, h, d_v] <- nm
          }
        }
      }

      # Output projection: out[q, d_out] = sum_h sum_dv hout[q,h,dv] * W_o(h,dv,d_out) + b_o
      out_nms <- character(0)
      for (q in seq_len(T_q)) {
        for (d_out in seq_len(C_out_w)) {
          terms <- c()
          for (h in seq_len(num_heads)) {
            for (d_v in seq_len(value_dim)) {
              terms <- c(
                terms,
                paste0(
                  backtick(hout_nm[q, h, d_v]),
                  " * ",
                  format_numeric(wo_at(h, d_v, d_out))
                )
              )
            }
          }
          nm <- paste0("orbital_mha_", lname, "_out_q", q, "_d", d_out)
          mha_nms <- c(mha_nms, nm)
          mha_exprs <- c(
            mha_exprs,
            paste0(
              "(",
              paste(terms, collapse = " + "),
              " + ",
              bo_at(d_out),
              ")"
            )
          )
          out_nms <- c(out_nms, nm)
        }
      }

      all_exprs[[lname]] <- stats::setNames(mha_exprs, mha_nms)
      assign(lname, out_nms, envir = expr_reg)
    } else {
      cli::cli_abort(c(
        "Unsupported layer type in Keras Functional model: {.cls {cls_orig}}.",
        "i" = paste(
          "orbital supports: Dense, EinsumDense, Add, Concatenate, BatchNormalization,",
          "LayerNormalization, InstanceNormalization, GroupNormalization,",
          "RMSNormalization, PReLU, GlobalAveragePooling1D, GlobalMaxPooling1D,",
          "AdaptiveAveragePooling1D, AdaptiveMaxPooling1D,",
          "AveragePooling1D, MaxPooling1D, GlobalSumPooling1D,",
          "Conv1D, ConvLSTM1D, LSTM, GRU, Bidirectional(LSTM/GRU), SimpleRNN,",
          "UnitNormalization, ZeroPadding1D, Embedding, TimeDistributed(Dense),",
          "MultiHeadAttention, Attention, AdditiveAttention, Subtract,",
          "UpSampling1D, Dropout, SpatialDropout1D, GaussianDropout,",
          "AlphaDropout, Flatten, Reshape, Activation, Softmax."
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
          "globalmaxpool|adaptiveaveragepooling1d|adaptivemaxpooling1d|averagepooling1d|maxpooling1d|globalsumpooling|",
          "\\brelu\\b|\\bactivation\\b|\\bsoftmax\\b|\\blstm\\b|\\bgru\\b|convlstm1d|conv1d|",
          "bidirectional|simplernn|unitnorm|zeropadding1d|multiheadattention|",
          "\\bmultiply\\b|\\baverage\\b|\\bmaximum\\b|\\bminimum\\b|\\bdot\\b|\\bsubtract\\b|",
          "permute|cropping1d|repeatvector|conv1dtranspose|\\battention\\b|additiveattention|",
          "depthwiseconv1d|separableconv1d|\\bembedding\\b|timedistributed|upsampling1d"
        ),
        cls,
        perl = TRUE
      ) &&
        !grepl(
          "dense|input|flatten|reshape|dropout|depthwiseconv2d|separableconv2d|2d|3d",
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
