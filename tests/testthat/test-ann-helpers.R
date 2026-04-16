test_that("activation_expr: linear returns x unchanged", {
  expr <- activation_expr("linear", "z")
  expect_equal(expr, "z")
})

test_that("activation_expr: relu returns max(0, x)", {
  expr <- activation_expr("relu", "z")
  expect_match(expr, "if_else")
  # relu(1) = 1, relu(-1) = 0
  df <- data.frame(z = c(-1, 0, 1, 2))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result, c(0, 0, 1, 2))
})

test_that("activation_expr: sigmoid maps to (0, 1)", {
  expr <- activation_expr("sigmoid", "z")
  expect_match(expr, "exp")
  df <- data.frame(z = c(-10, 0, 10))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result, c(0, 0.5, 1), tolerance = 1e-4)
})

test_that("activation_expr: tanh produces correct values", {
  expr <- activation_expr("tanh", "z")
  expect_match(expr, "tanh")
  df <- data.frame(z = c(-1, 0, 1))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result, tanh(c(-1, 0, 1)), tolerance = 1e-6)
})

test_that("activation_expr: elu is identity for positive inputs", {
  expr <- suppressWarnings(activation_expr("elu", "z"))
  expect_match(expr, "if_else")
  df <- data.frame(z = c(-1, 0, 1, 2))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result[3:4], c(1, 2))
  # elu(-1) = exp(-1) - 1 ≈ -0.6321
  expect_equal(result[1], exp(-1) - 1, tolerance = 1e-6)
})

test_that("activation_expr: celu is identity for positive inputs", {
  expr <- suppressWarnings(activation_expr("celu", "z"))
  expect_match(expr, "if_else")
  df <- data.frame(z = c(-1, 0, 2))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result[3], 2)
  # default alpha = 1 → celu(-1) = -1 * (exp(1) - 1)  ... wait: exp(-1/1)-1 = exp(-1)-1
  # celu(x, alpha=1) = max(0,x) + min(0, alpha*(exp(x/alpha)-1))
  expect_equal(result[1], exp(-1) - 1, tolerance = 1e-6)
})

test_that("activation_expr: selu uses correct scale constants", {
  expr <- activation_expr("selu", "z")
  expect_match(expr, "1.0507009873554805")
  # positive: gamma * x
  df <- data.frame(z = c(-2, 0, 1, 2))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  gamma <- 1.0507009873554805
  alpha <- 1.6732632423543772
  expect_equal(result[3], gamma * 1, tolerance = 1e-7)
  expect_equal(result[4], gamma * 2, tolerance = 1e-7)
  expect_equal(result[1], gamma * alpha * (exp(-2) - 1), tolerance = 1e-7)
})

test_that("activation_expr: gelu produces correct values", {
  expr <- activation_expr("gelu", "z")
  # activation_expr("gelu") uses the exact-erf form (A&S 7.1.28 approximation).
  # The expression contains the 1/sqrt(2) scale constant for the erf argument.
  expect_match(expr, "0.7071067811865476")
  df <- data.frame(z = c(0, 1, -1))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  # Reference values from the exact GELU definition using base-R pnorm:
  # gelu(z) = z * 0.5 * (1 + erf(z / sqrt(2)))
  # erf(z/sqrt(2)) = 2*pnorm(z) - 1  (since pnorm(x) = 0.5*(1 + erf(x/sqrt(2))))
  gelu_exact_ref <- function(z) {
    erf_z <- 2 * pnorm(z) - 1
    z * 0.5 * (1 + erf_z)
  }
  # Tolerance 1e-5 covers the A&S polynomial approximation error (~1.5e-7)
  expect_equal(result, gelu_exact_ref(c(0, 1, -1)), tolerance = 1e-5)
  # Qualitative checks: gelu(0)=0, gelu(1)>0, gelu(-1)<0
  expect_equal(result[1], 0, tolerance = 1e-8)
  expect_true(result[2] > 0.8)
  expect_true(result[3] < 0)
})

test_that("activation_expr: hardtanh clamps to [-1, 1]", {
  expr <- activation_expr("hardtanh", "z")
  df <- data.frame(z = c(-5, -1, 0, 1, 5))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result, c(-1, -1, 0, 1, 1))
})

test_that("activation_expr: hardsigmoid (PyTorch ±3 clamp) correct values", {
  expr <- activation_expr("hardsigmoid", "z")
  df <- data.frame(z = c(-4, -3, 0, 3, 4))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result, c(0, 0, 0.5, 1, 1), tolerance = 1e-8)
})

test_that("activation_expr: hard_sigmoid (Keras3 ±3 clamp) correct values", {
  expr <- activation_expr("hard_sigmoid", "z")
  df <- data.frame(z = c(-4, -3, 0, 3, 4))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result, c(0, 0, 0.5, 1, 1), tolerance = 1e-8)
})

test_that("activation_expr: hard_swish (Keras3) correct values", {
  expr <- activation_expr("hard_swish", "z")
  df <- data.frame(z = c(-4, -3, 0, 3, 4))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  # x * hard_sigmoid(x) with ±3 gate
  expect_equal(result[1], 0, tolerance = 1e-8) # below gate
  expect_equal(result[5], 4, tolerance = 1e-8) # above gate
  expect_equal(result[3], 0, tolerance = 1e-8) # x=0 → 0 * 0.5 = 0
})

test_that("activation_expr: leaky_relu uses correct negative slope", {
  expr <- suppressWarnings(activation_expr("leaky_relu", "z"))
  expect_match(expr, "0.01|if_else")
  df <- data.frame(z = c(-2, 0, 2))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result[1], -0.02, tolerance = 1e-8)
  expect_equal(result[3], 2, tolerance = 1e-8)
})

test_that("activation_expr: log_sigmoid returns -log(1+exp(-x))", {
  expr <- activation_expr("log_sigmoid", "z")
  expect_match(expr, "log")
  df <- data.frame(z = c(-1, 0, 1))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expected <- -log(1 + exp(-c(-1, 0, 1)))
  expect_equal(result, expected, tolerance = 1e-6)
})

test_that("activation_expr: softplus returns log(1 + exp(x))", {
  expr <- activation_expr("softplus", "z")
  df <- data.frame(z = c(-1, 0, 1))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result, log(1 + exp(c(-1, 0, 1))), tolerance = 1e-6)
})

test_that("activation_expr: mish returns x * tanh(softplus(x))", {
  expr <- activation_expr("mish", "z")
  expect_match(expr, "tanh")
  expect_match(expr, "log")
  df <- data.frame(z = c(-2, 0, 2))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expected <- c(-2, 0, 2) * tanh(log(1 + exp(c(-2, 0, 2))))
  expect_equal(result, expected, tolerance = 1e-6)
})

test_that("activation_expr: softsign returns x / (1 + |x|)", {
  expr <- activation_expr("softsign", "z")
  expect_match(expr, "abs")
  df <- data.frame(z = c(-1, 0, 2))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result, c(-0.5, 0, 2 / 3), tolerance = 1e-8)
})

test_that("activation_expr: swish / silu returns x * sigmoid(x)", {
  expr_swish <- activation_expr("swish", "z")
  expr_silu <- activation_expr("silu", "z")
  df <- data.frame(z = c(-1, 0, 1))
  r_swish <- dplyr::mutate(df, r = !!rlang::parse_expr(expr_swish))$r
  r_silu <- dplyr::mutate(df, r = !!rlang::parse_expr(expr_silu))$r
  expected <- c(-1, 0, 1) / (1 + exp(-c(-1, 0, 1)))
  expect_equal(r_swish, expected, tolerance = 1e-6)
  expect_equal(r_silu, expected, tolerance = 1e-6)
})

test_that("activation_expr: relu6 clamps at 6", {
  expr <- activation_expr("relu6", "z")
  df <- data.frame(z = c(-1, 0, 3, 6, 7))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result, c(0, 0, 3, 6, 6))
})

test_that("activation_expr: hardshrink zeroes values inside threshold ±0.5", {
  expr <- activation_expr("hardshrink", "z")
  df <- data.frame(z = c(-1, -0.3, 0, 0.3, 1))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result, c(-1, 0, 0, 0, 1))
})

test_that("activation_expr: softshrink shifts values outside ±0.5 toward zero", {
  expr <- activation_expr("softshrink", "z")
  df <- data.frame(z = c(-2, -0.3, 0, 0.3, 2))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result[1], -1.5, tolerance = 1e-8)
  expect_equal(result[5], 1.5, tolerance = 1e-8)
  expect_equal(result[2], 0, tolerance = 1e-8)
  expect_equal(result[3], 0, tolerance = 1e-8)
})

test_that("activation_expr: tanhshrink returns x - tanh(x)", {
  expr <- activation_expr("tanhshrink", "z")
  df <- data.frame(z = c(-1, 0, 1))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result, c(-1, 0, 1) - tanh(c(-1, 0, 1)), tolerance = 1e-6)
})

test_that("activation_expr: gelu_approximate (tanh-approx GELU) correct values", {
  expr_approx <- activation_expr("gelu_approximate", "z")
  expr_tanh <- activation_expr("gelu_tanh", "z")
  # Both aliases must produce the same expression
  expect_equal(expr_approx, expr_tanh)
  # The expression must contain the sqrt(2/pi) constant
  expect_match(expr_approx, "0.7978845608028654")
  df <- data.frame(z = c(0, 1, -1, 2))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr_approx))$r
  # tanh-approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
  gelu_tanh_ref <- function(z) {
    z * 0.5 * (1 + tanh(0.7978845608028654 * (z + 0.044715 * z^3)))
  }
  expect_equal(result, gelu_tanh_ref(c(0, 1, -1, 2)), tolerance = 1e-7)
  # gelu_approximate(0) = 0
  expect_equal(result[1], 0, tolerance = 1e-9)
})

test_that("activation_expr: rrelu uses fixed mid-point slope 11/48", {
  expr <- activation_expr("rrelu", "z")
  expect_match(expr, "0.22916666666666666")
  df <- data.frame(z = c(-2, 0, 3))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  # positive: passthrough; negative: slope = 11/48 ≈ 0.229166…
  expect_equal(result[3], 3, tolerance = 1e-9)
  expect_equal(result[1], -2 * (11 / 48), tolerance = 1e-8)
  expect_equal(result[2], 0, tolerance = 1e-9)
})

test_that("activation_expr: log_softmax raises informative error (requires DAG path)", {
  expect_error(
    activation_expr("log_softmax", "z"),
    regexp = "log_softmax"
  )
})

test_that("activation_expr: sparsemax raises informative error (requires DAG path)", {
  expect_error(
    activation_expr("sparsemax", "z"),
    regexp = "sparsemax"
  )
})

test_that("activation_expr: unknown activation raises an error", {
  expect_error(activation_expr("nonexistent_act_xyz", "z"))
})

# ── build_mlp_pre_act ─────────────────────────────────────────────────────────

test_that("build_mlp_pre_act skips zero weights and includes non-zero weights", {
  # Diagonal weight matrix: output 1 uses only x1, output 2 uses only x2
  wmat <- matrix(c(1.0, 0.0, 0.0, 2.0), nrow = 2, ncol = 2)
  biases <- c(0.5, -0.5)
  input_names <- c("x1", "x2")
  result <- build_mlp_pre_act(wmat, biases, input_names)
  expect_length(result, 2)
  # Output 1 should mention x1 but not x2
  expect_match(result[1], "x1")
  expect_false(grepl("x2", result[1]))
  # Output 2 should mention x2 but not x1
  expect_match(result[2], "x2")
  expect_false(grepl("x1", result[2]))
})

test_that("build_mlp_pre_act includes bias even when all weights are zero", {
  wmat <- matrix(0, nrow = 1, ncol = 2)
  biases <- 3.14
  input_names <- c("a", "b")
  result <- build_mlp_pre_act(wmat, biases, input_names)
  expect_length(result, 1)
  # Only the bias term should remain; no input variable references
  expect_false(grepl("\\ba\\b|\\bb\\b", result[1]))
})

test_that("activation_expr: exponential returns exp(x)", {
  expr <- activation_expr("exponential", "z")
  expect_match(expr, "exp")
  df <- data.frame(z = c(-1, 0, 1, 2))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result, exp(c(-1, 0, 1, 2)), tolerance = 1e-8)
})

test_that("activation_expr: threshold zeroes values at or below threshold", {
  expr <- suppressWarnings(activation_expr("threshold", "z"))
  # Default threshold = 1.0, default value = 0
  expect_match(expr, "if_else")
  df <- data.frame(z = c(-1, 0, 1, 1.5, 2))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  # x > 1 → passthrough; x <= 1 → 0
  expect_equal(result, c(0, 0, 0, 1.5, 2))
})

test_that("activation_expr: threshold respects custom alpha", {
  expr <- activation_expr("threshold", "z", alpha = 2.0)
  df <- data.frame(z = c(-1, 1, 2, 2.5, 3))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  # x > 2 → passthrough; x <= 2 → 0
  expect_equal(result, c(0, 0, 0, 2.5, 3))
})

test_that("activation_expr: hard_silu is alias for hard_swish", {
  expr_hard_silu <- activation_expr("hard_silu", "z")
  expr_hard_swish <- activation_expr("hard_swish", "z")
  expect_equal(expr_hard_silu, expr_hard_swish)
  df <- data.frame(z = c(-4, 0, 4))
  r_hard_silu <- dplyr::mutate(df, r = !!rlang::parse_expr(expr_hard_silu))$r
  r_hard_swish <- dplyr::mutate(df, r = !!rlang::parse_expr(expr_hard_swish))$r
  expect_equal(r_hard_silu, r_hard_swish, tolerance = 1e-8)
})

# ── alpha fallback warnings (R-A: cli_warn hardening) ────────────────────────

test_that("activation_expr: elu warns with orbital_alpha_default when alpha is NULL", {
  expect_warning(
    activation_expr("elu", "z", alpha = NULL),
    class = "orbital_alpha_default"
  )
})

test_that("activation_expr: elu uses default alpha 1.0 when alpha is NULL", {
  expr <- suppressWarnings(activation_expr("elu", "z", alpha = NULL))
  df <- data.frame(z = c(-1, 0, 1))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  # default alpha = 1.0: elu(-1) = exp(-1) - 1
  expect_equal(result[1], exp(-1) - 1, tolerance = 1e-8)
  expect_equal(result[3], 1, tolerance = 1e-8)
})

test_that("activation_expr: elu does not warn when alpha is provided", {
  expect_no_warning(activation_expr("elu", "z", alpha = 0.5))
})

test_that("activation_expr: celu warns with orbital_alpha_default when alpha is NULL", {
  expect_warning(
    activation_expr("celu", "z", alpha = NULL),
    class = "orbital_alpha_default"
  )
})

test_that("activation_expr: celu does not warn when alpha is provided", {
  expect_no_warning(activation_expr("celu", "z", alpha = 1.0))
})

test_that("activation_expr: leaky_relu warns with orbital_alpha_default when alpha is NULL", {
  expect_warning(
    activation_expr("leaky_relu", "z", alpha = NULL),
    class = "orbital_alpha_default"
  )
})

test_that("activation_expr: leaky_relu default alpha 0.01 gives correct values", {
  expr <- suppressWarnings(activation_expr("leaky_relu", "z", alpha = NULL))
  df <- data.frame(z = c(-2, 0, 2))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result[1], -0.02, tolerance = 1e-8)
  expect_equal(result[3], 2, tolerance = 1e-8)
})

test_that("activation_expr: leaky_relu does not warn when alpha is provided", {
  expect_no_warning(activation_expr("leaky_relu", "z", alpha = 0.2))
})

test_that("activation_expr: threshold warns with orbital_alpha_default when alpha is NULL", {
  expect_warning(
    activation_expr("threshold", "z", alpha = NULL),
    class = "orbital_alpha_default"
  )
})

test_that("activation_expr: threshold default alpha 1.0 gives correct values", {
  expr <- suppressWarnings(activation_expr("threshold", "z", alpha = NULL))
  df <- data.frame(z = c(-1, 1, 1.5, 2))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  # x > 1.0 passes through; x <= 1.0 → 0
  expect_equal(result, c(0, 0, 1.5, 2))
})

test_that("activation_expr: threshold does not warn when alpha is provided", {
  expect_no_warning(activation_expr("threshold", "z", alpha = 2.0))
})

# ── Wave 1A regression: hard_sigmoid formula fix ──────────────────────────────

test_that("hard_sigmoid and hardsigmoid produce identical expression strings (Wave 1A regression)", {
  expr_hs <- activation_expr("hard_sigmoid", "z")
  expr_hsi <- activation_expr("hardsigmoid", "z")
  expect_equal(expr_hs, expr_hsi)
})

test_that("hard_sigmoid uses ±3 clamp and x/6 slope — old ±2.5/0.2*x constants absent (Wave 1A regression)", {
  expr <- activation_expr("hard_sigmoid", "z")
  # New constants must be present
  expect_true(grepl("/ 6", expr, fixed = TRUE))
  expect_true(grepl("0.5", expr, fixed = TRUE))
  # Old wrong constants must NOT appear
  expect_false(grepl("2\\.5", expr))
  expect_false(grepl("0\\.2", expr))
})

test_that("hard_sigmoid numerical values match ±3 clamp spec at all critical points (Wave 1A regression)", {
  expr <- activation_expr("hard_sigmoid", "z")
  zvals <- c(-4, -3, -1.5, 0, 1.5, 3, 4)
  df <- data.frame(z = zvals)
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  # z <= -3 → 0, z >= 3 → 1, otherwise x/6 + 0.5
  expected <- ifelse(zvals <= -3, 0, ifelse(zvals >= 3, 1, zvals / 6 + 0.5))
  expect_equal(result, expected, tolerance = 1e-8)
})

# ── Wave 1A regression: activation name aliases ────────────────────────────────

test_that("hard_shrink is an alias for hardshrink — produces identical expression (Wave 1A regression)", {
  expect_equal(
    activation_expr("hard_shrink", "z"),
    activation_expr("hardshrink", "z")
  )
})

test_that("soft_shrink is an alias for softshrink — produces identical expression (Wave 1A regression)", {
  expect_equal(
    activation_expr("soft_shrink", "z"),
    activation_expr("softshrink", "z")
  )
})

test_that("tanh_shrink is an alias for tanhshrink — produces identical expression (Wave 1A regression)", {
  expect_equal(
    activation_expr("tanh_shrink", "z"),
    activation_expr("tanhshrink", "z")
  )
})

# ── Wave 1A regression: threshold default_value parameter ─────────────────────

test_that("threshold with default_value=-1.5 embeds -1.5 in expression (Wave 1A regression)", {
  expr <- activation_expr("threshold", "z", alpha = 0.5, default_value = -1.5)
  expect_true(grepl("-1.5", expr, fixed = TRUE))
  # Numerically: z > 0.5 → z, else -1.5
  df <- data.frame(z = c(0, 0.5, 1, 2))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result, c(-1.5, -1.5, 1, 2), tolerance = 1e-8)
})

test_that("threshold default_value=0 matches the no-default-value call (Wave 1A regression)", {
  expr0 <- activation_expr("threshold", "z", alpha = 1.0, default_value = 0)
  exprDflt <- suppressWarnings(activation_expr("threshold", "z"))
  df <- data.frame(z = c(-1, 0, 0.5, 1, 2))
  r0 <- dplyr::mutate(df, r = !!rlang::parse_expr(expr0))$r
  rD <- dplyr::mutate(df, r = !!rlang::parse_expr(exprDflt))$r
  expect_equal(r0, rD, tolerance = 1e-9)
})

# ── prelu ─────────────────────────────────────────────────────────────────────

test_that("activation_expr: prelu with explicit alpha uses per-channel slope", {
  expr <- activation_expr("prelu", "z", alpha = 0.1)
  df <- data.frame(z = c(-2, 0, 2))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result[1], -0.2, tolerance = 1e-8) # 0.1 * -2
  expect_equal(result[2], 0, tolerance = 1e-8)
  expect_equal(result[3], 2, tolerance = 1e-8)
})

test_that("activation_expr: prelu warns with orbital_alpha_default when alpha is NULL", {
  expect_warning(
    activation_expr("prelu", "z", alpha = NULL),
    class = "orbital_alpha_default"
  )
})

test_that("activation_expr: prelu default alpha 0.25 gives correct values", {
  expr <- suppressWarnings(activation_expr("prelu", "z", alpha = NULL))
  df <- data.frame(z = c(-4, 0, 4))
  result <- dplyr::mutate(df, r = !!rlang::parse_expr(expr))$r
  expect_equal(result[1], -1.0, tolerance = 1e-8) # 0.25 * -4
  expect_equal(result[3], 4, tolerance = 1e-8)
})

test_that("activation_expr: prelu does not warn when alpha is provided", {
  expect_no_warning(activation_expr("prelu", "z", alpha = 0.5))
})

# ── softmax error paths ───────────────────────────────────────────────────────

test_that("activation_expr: softmax raises informative error (requires DAG path)", {
  expect_error(
    activation_expr("softmax", "z"),
    regexp = "softmax"
  )
})
