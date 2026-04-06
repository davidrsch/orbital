# Handler functions for normalisation and activation layers
# (BatchNorm, LayerNorm, PReLU, Dropout pass-through, LeakyReLU, ELU, ReLU,
#  Activation, InstanceNorm, GroupNorm, RMSNorm, Softmax, UnitNorm).
# Called by orbital_keras_dag_impl() in model-keras.R.


.k3_batchnorm <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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
  state$all_exprs[[lname]] <- stats::setNames(bn_exprs, unit_names)
  assign(lname, unit_names, envir = expr_reg)
  invisible(NULL)
}


.k3_layernorm <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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
  state$all_exprs[[lname]] <- c(
    stats::setNames(mean_expr, mean_nm),
    stats::setNames(var_expr, var_nm),
    stats::setNames(norm_exprs, unit_names)
  )
  assign(lname, unit_names, envir = expr_reg)
  invisible(NULL)
}


.k3_prelu <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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
  state$all_exprs[[lname]] <- stats::setNames(prelu_exprs, unit_names)
  assign(lname, unit_names, envir = expr_reg)
  invisible(NULL)
}


.k3_dropout_passthru <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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
  invisible(NULL)
}


.k3_leakyrelu <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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
  state$all_exprs[[lname]] <- stats::setNames(lr_exprs, unit_names)
  assign(lname, unit_names, envir = expr_reg)
  invisible(NULL)
}


.k3_elu <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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
  state$all_exprs[[lname]] <- stats::setNames(elu_exprs, unit_names)
  assign(lname, unit_names, envir = expr_reg)
  invisible(NULL)
}


.k3_relu <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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
  state$all_exprs[[lname]] <- stats::setNames(relu_exprs, unit_names)
  assign(lname, unit_names, envir = expr_reg)
  invisible(NULL)
}


.k3_activation <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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
    state$all_exprs[[lname]] <- c(
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
    state$all_exprs[[lname]] <- stats::setNames(act_exprs, unit_names)
    assign(lname, unit_names, envir = expr_reg)
  }
  invisible(NULL)
}


.k3_instancenorm <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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
  state$all_exprs[[lname]] <- c(
    stats::setNames(mean_expr, mean_nm),
    stats::setNames(var_expr, var_nm),
    stats::setNames(norm_exprs, unit_names)
  )
  assign(lname, unit_names, envir = expr_reg)
  invisible(NULL)
}


.k3_groupnorm <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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
    state$all_exprs[[lname]] <- c(
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
    state$all_exprs[[lname]] <- stats::setNames(
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

    state$all_exprs[[lname]] <- stats::setNames(
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
  invisible(NULL)
}


.k3_rmsnorm <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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
  state$all_exprs[[lname]] <- c(
    stats::setNames(rms_expr, rms_nm),
    stats::setNames(norm_exprs, unit_names)
  )
  assign(lname, unit_names, envir = expr_reg)
  invisible(NULL)
}


.k3_softmax <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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
  state$all_exprs[[lname]] <- c(
    stats::setNames(sm_max_expr, sm_max_nm),
    stats::setNames(sm_sum_expr, sm_sum_nm),
    stats::setNames(sm_exprs, unit_names)
  )
  assign(lname, unit_names, envir = expr_reg)
  invisible(NULL)
}


.k3_unitnorm <- function(l, lname, topo_map, expr_reg, state, weight_map, output_layer_names, last_dense) {
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
  state$all_exprs[[lname]] <- c(
    stats::setNames(norm_expr, norm_nm),
    stats::setNames(unit_exprs, unit_names)
  )
  assign(lname, unit_names, envir = expr_reg)
  invisible(NULL)
}

