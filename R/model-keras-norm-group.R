.k3_instancenorm <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
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
        "(({xe} - `{mean_nm}`) / sqrt(`{var_nm}` + {format_numeric(eps)}))",
        " * {format_numeric(gamma[i])} + {format_numeric(beta[i])}"
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


.k3_groupnorm <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
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
          "(({xe} - `{mean_nm}`) / sqrt(`{var_nm}` + {format_numeric(eps)}))",
          " * {format_numeric(gamma[i])} + {format_numeric(beta[i])}"
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
      norm_exprs_g <- local({
        .mean_nm <- mean_nm
        .var_nm <- var_nm
        .start <- start
        .in_exprs <- in_exprs
        .gamma <- gamma
        .beta <- beta
        .eps <- eps
        .unit_names_g <- unit_names_g
        vapply(
          seq_along(.unit_names_g),
          function(i_local) {
            c_idx <- .start + i_local - 1L
            xe <- backtick(.in_exprs[[c_idx]])
            glue::glue(
              "(({xe} - `{.mean_nm}`) / sqrt(`{.var_nm}` + {format_numeric(.eps)}))",
              " * {format_numeric(.gamma[c_idx])} + {format_numeric(.beta[c_idx])}"
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
      "Unsupported GroupNormalization configuration in Keras model: {.val {lname}}.",
      "i" = "has {num_groups} groups for {n_feat} features.",
      "i" = "num_groups must evenly divide the number of features ({n_feat} %% {num_groups} != 0)."
    ))
  }
  invisible(NULL)
}


.k3_rmsnorm <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # RMSNormalization: per-row normalize using root mean square (no mean subtraction).
  # rms  = sqrt((1/C) * sum_c(x_c^2) + epsilon)
  # y_c  = (x_c / rms) * scale_c
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  wts <- l$get_weights() # scale only (no bias)
  gamma <- as.numeric(wts[[1L]])
  n_feat <- length(in_exprs)
  eps <- tryCatch(as.numeric(l$epsilon), error = function(e) 1e-8)
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


.k3_softmax <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
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
