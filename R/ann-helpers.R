# Helper functions for artificial neural network (MLP) model implementations
# Used by model-nnet.R, model-brulee.R, model-keras.R, model-h2o.R
# kerasnip dispatch (unwrapping list fit slot) is handled in parsnip.R

# Build an inline dplyr/SQL-portable expression string for erf(z_expr).
# Uses the Abramowitz & Stegun (1964) §7.1.28 polynomial approximation;
# maximum absolute error < 1.5e-7.
# This mirrors the Python orbital._erf_approx() helper in erf.py.
.erf_approx_expr <- function(z_expr) {
  t <- glue::glue("(1.0 / (1.0 + 0.3275911 * abs({z_expr})))")
  poly <- glue::glue(
    "({t} * (0.254829592 + {t} * (-0.284496736 + {t} * (1.421413741 + {t} * (-1.453152027 + {t} * 1.061405429)))))"
  )
  glue::glue(
    "dplyr::if_else(({z_expr}) >= 0.0, 1.0, -1.0) * (1.0 - {poly} * exp(-({z_expr})^2))"
  )
}

# Build an inline expression for GELU using the exact erf form:
# gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
# where 1/sqrt(2) = 0.7071067811865476.
.gelu_exact_expr <- function(x_expr) {
  z <- glue::glue("({x_expr}) * 0.7071067811865476")
  erf_z <- .erf_approx_expr(z)
  glue::glue("({x_expr}) * 0.5 * (1.0 + ({erf_z}))")
}

activation_expr <- function(activation, x_expr, alpha = NULL) {
  switch(
    activation,
    "linear" = as.character(x_expr),
    "relu" = ,
    "Rectifier" = ,
    "RectifierWithDropout" = glue::glue(
      "dplyr::if_else({x_expr} > 0, {x_expr}, 0)"
    ),
    "sigmoid" = glue::glue("1 / (1 + exp(-({x_expr})))"),
    "tanh" = ,
    "Tanh" = ,
    "TanhWithDropout" = glue::glue("tanh({x_expr})"),
    "elu" = {
      a <- if (is.null(alpha)) 1.0 else alpha
      glue::glue(
        "dplyr::if_else({x_expr} >= 0, {x_expr}, {format_numeric(a)} * (exp({x_expr}) - 1))"
      )
    },
    "celu" = {
      a <- if (is.null(alpha)) 1.0 else alpha
      glue::glue(
        "dplyr::if_else({x_expr} >= 0, {x_expr}, {format_numeric(a)} * (exp({x_expr} / {format_numeric(a)}) - 1))"
      )
    },
    # SELU constants from Klambauer et al. 2017 (https://arxiv.org/abs/1706.02515):
    #   SELU_ALPHA  = 1.6732632423543772
    #   SELU_GAMMA  = 1.0507009873554805
    #   SELU_GAMMA * SELU_ALPHA = 1.7580993408474319
    # Note: PyTorch uses alpha=1.6732631921768192 (differs by ~5e-8), which is
    # the value present in the ONNX spec. Keras 3 uses the paper value above.
    "selu" = glue::glue(
      "dplyr::if_else({x_expr} > 0, 1.0507009873554805 * {x_expr}, 1.7580993408474319 * (exp({x_expr}) - 1))"
    ),
    # Exact GELU (default in Keras3 / PyTorch approximate=False):
    # gelu(x) = x * 0.5 * (1 + erf(x/sqrt(2))) via A&S 7.1.28 polynomial.
    "gelu" = .gelu_exact_expr(x_expr),
    # Tanh-approximation GELU (PyTorch approximate="tanh" / brulee):
    # gelu_approx(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    "gelu_approximate" = ,
    "gelu_tanh" = glue::glue(
      "{x_expr} * 0.5 * (1 + tanh(({x_expr} + 0.044715 * {x_expr}^3) * 0.7978845608028654))"
    ),
    "hardshrink" = glue::glue(
      "dplyr::if_else(abs({x_expr}) > 0.5, {x_expr}, 0)"
    ),
    # PyTorch hardsigmoid: clamp x to [-3, 3], then x/6 + 0.5
    "hardsigmoid" = glue::glue(
      "dplyr::if_else({x_expr} <= -3, 0, dplyr::if_else({x_expr} >= 3, 1, {x_expr} / 6 + 0.5))"
    ),
    # Keras3 hard_sigmoid: clamp x to [-2.5, 2.5], then 0.2*x + 0.5
    "hard_sigmoid" = glue::glue(
      "dplyr::if_else({x_expr} <= -2.5, 0, dplyr::if_else({x_expr} >= 2.5, 1, 0.2 * {x_expr} + 0.5))"
    ),
    "hardtanh" = glue::glue(
      "dplyr::if_else({x_expr} < -1, -1, dplyr::if_else({x_expr} > 1, 1, {x_expr}))"
    ),
    "hard_silu" = ,
    "hard_swish" = ,
    "hardswish" = glue::glue(
      "{x_expr} * dplyr::if_else({x_expr} <= -3, 0, dplyr::if_else({x_expr} >= 3, 1, ({x_expr} + 3) / 6))"
    ),
    "leaky_relu" = {
      a <- if (is.null(alpha)) 0.01 else alpha
      glue::glue(
        "dplyr::if_else({x_expr} >= 0, {x_expr}, {format_numeric(a)} * {x_expr})"
      )
    },
    "log_sigmoid" = glue::glue("-log(1 + exp(-({x_expr})))"),
    "relu6" = glue::glue(
      "dplyr::if_else({x_expr} < 0, 0, dplyr::if_else({x_expr} > 6, 6, {x_expr}))"
    ),
    "rrelu" = glue::glue(
      "dplyr::if_else({x_expr} >= 0, {x_expr}, 0.22916666666666666 * {x_expr})"
    ),
    "silu" = ,
    "swish" = glue::glue("{x_expr} * (1 / (1 + exp(-({x_expr}))))"),
    "exponential" = glue::glue("exp({x_expr})"),
    "threshold" = {
      theta <- if (is.null(alpha)) 1.0 else alpha
      glue::glue(
        "dplyr::if_else({x_expr} > {format_numeric(theta)}, {x_expr}, 0)"
      )
    },
    "softplus" = glue::glue("log(1 + exp({x_expr}))"),
    "mish" = glue::glue("{x_expr} * tanh(log(1 + exp({x_expr})))"),
    "softshrink" = glue::glue(
      "dplyr::if_else({x_expr} > 0.5, {x_expr} - 0.5, dplyr::if_else({x_expr} < -0.5, {x_expr} + 0.5, 0))"
    ),
    "softsign" = glue::glue("{x_expr} / (1 + abs({x_expr}))"),
    "tanhshrink" = glue::glue("{x_expr} - tanh({x_expr})"),
    # NOTE: "softmax" and "log_softmax" are intentionally omitted here.
    # Both functions normalise across *all* units simultaneously and therefore
    # cannot be expressed as independent per-unit scalar expressions.
    # They are handled structurally in the DAG path (orbital_keras_dag_impl)
    # via a dedicated softmax / log_softmax block that builds the shared
    # sum-of-exponentials intermediate expression.  Passing either name to
    # this function from Dense-layer activation strings raises an informative
    # error to guide the caller toward the correct layer-level approach.
    "softmax" = cli::cli_abort(
      c(
        "Activation {.val softmax} cannot be applied as a per-unit scalar expression.",
        "i" = paste(
          "softmax normalises across all units simultaneously.",
          "Use a standalone {.cls Activation(\"softmax\")} layer or a",
          "standalone {.cls Softmax} layer placed after a linear Dense layer."
        ),
        "i" = "The DAG path in orbital handles standalone Softmax layers and Activation('softmax') correctly."
      )
    ),
    "log_softmax" = cli::cli_abort(
      c(
        "Activation {.val log_softmax} cannot be applied as a per-unit scalar expression.",
        "i" = paste(
          "log_softmax normalises across all units simultaneously.",
          "Use a standalone {.cls Activation(\"log_softmax\")} layer",
          "placed after a linear Dense layer."
        ),
        "i" = "The DAG path in orbital handles standalone Activation layers with softmax/log_softmax correctly."
      )
    ),
    # sparsemax normalises like softmax (projects onto the probability simplex)
    # and therefore also requires structural/DAG-level handling.
    "sparsemax" = cli::cli_abort(
      c(
        "Activation {.val sparsemax} cannot be applied as a per-unit scalar expression.",
        "i" = paste(
          "sparsemax normalises across all units simultaneously",
          "(projects each row onto the probability simplex).",
          "It requires structural handling similar to softmax."
        )
      )
    ),
    cli::cli_abort(
      "Activation function {.val {activation}} is not supported by orbital."
    )
  )
}

build_mlp_pre_act <- function(weight_mat, biases, input_names) {
  n_out <- nrow(weight_mat)
  vapply(
    seq_len(n_out),
    function(i) {
      nz_idx <- which(weight_mat[i, ] != 0)
      wterms <- vapply(
        nz_idx,
        function(j) {
          paste0(
            "(",
            backtick(input_names[j]),
            " * ",
            format_numeric(weight_mat[i, j]),
            ")"
          )
        },
        character(1)
      )
      paste(c(format_numeric(biases[i]), wterms), collapse = " + ")
    },
    character(1)
  )
}
