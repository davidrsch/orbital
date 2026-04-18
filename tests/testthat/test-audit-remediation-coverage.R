# Audit-remediation tests:
#   M7  - PReLU per-unit alpha branch in orbital_brulee_mlp_impl()
#   M9  - shrink-family / niche activation_expr() expressions are well-formed
#   M10 - brulee_two_layer activation_2 with various non-softmax values
#   L7  - brulee regression with extreme y_stats (huge mean / tiny sd)

# ---------- M9: shrink-family / niche activation_expr() expressions ----------

test_that("M9: shrink-family and niche activations emit valid SQL expressions", {
  x <- "z"
  cases <- list(
    hardshrink = "if_else",
    soft_shrink = "if_else",
    softshrink = "if_else",
    hard_shrink = "if_else",
    tanh_shrink = "tanh",
    tanhshrink = "tanh",
    log_sigmoid = "log",
    exponential = "exp",
    rrelu = "if_else",
    squareplus = "sqrt",
    softplus = "log1p"
  )

  for (act in names(cases)) {
    expr <- activation_expr(act, x)
    expect_type(expr, "character")
    expect_true(
      nzchar(expr),
      info = paste("activation =", act)
    )
    expect_true(
      grepl(cases[[act]], expr, fixed = TRUE),
      info = paste(
        "activation =",
        act,
        "expected token",
        cases[[act]],
        "got:",
        expr
      )
    )
    # The pre-activation variable name must appear in the emitted expression.
    expect_true(
      grepl(x, expr, fixed = TRUE),
      info = paste("activation =", act)
    )
  }
})

test_that("M9: threshold activation honours alpha and default_value", {
  expr <- activation_expr("threshold", "z", alpha = 2.0, default_value = -1)
  expect_true(grepl("> 2", expr, fixed = TRUE))
  expect_true(grepl("-1", expr, fixed = TRUE))
})

# ---------- M7: PReLU per-unit alpha branch ----------

test_that("M7: PReLU with per-unit alphas emits per-unit slopes", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("brulee")
  skip_if(!torch::torch_is_installed(), "torch not installed")

  spec <- parsnip::mlp(hidden_units = 4, epochs = 10, engine = "brulee") |>
    parsnip::set_mode("regression") |>
    parsnip::set_engine("brulee", activation = "prelu")

  set.seed(1)
  fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit)

  # Predictions match brulee's own predict() — this is only true when the
  # per-unit (length(alpha) > 1) PReLU branch in orbital_brulee_mlp_impl()
  # uses the correct per-channel slopes.
  preds <- predict(orb_obj, mtcars)
  exps <- predict(fit, mtcars)
  exps <- as.data.frame(exps)
  rownames(preds) <- NULL
  rownames(exps) <- NULL
  expect_equal(preds, exps, tolerance = 1e-5)

  # The emitted expressions should reference *distinct* PReLU slopes.
  exprs <- vapply(orb_obj, as.character, character(1L))
  hidden_exprs <- exprs[grepl("orbital_mlp_l1_h", names(exprs))]
  # Look for floating-point literals in the if_else negative branch;
  # if all branches share the same slope the per-unit code path was not
  # exercised. We just assert that >= 2 distinct numeric literals appear
  # across the four hidden units' if_else negative branches.
  neg_slopes <- regmatches(
    hidden_exprs,
    gregexpr("-?[0-9]+\\.[0-9]+", hidden_exprs, perl = TRUE)
  )
  flat <- unique(unlist(neg_slopes))
  expect_gte(length(flat), 2L)
})

# ---------- M10: brulee_two_layer activation_2 non-softmax sweep ----------

test_that("M10: brulee_two_layer with activation_2 sweep matches predictions", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("brulee")
  skip_if(!torch::torch_is_installed(), "torch not installed")

  for (act2 in c("tanh", "sigmoid", "linear", "elu", "celu")) {
    spec <- parsnip::mlp(
      hidden_units = 4,
      epochs = 10,
      engine = "brulee_two_layer"
    ) |>
      parsnip::set_mode("regression") |>
      parsnip::set_engine(
        "brulee_two_layer",
        hidden_units_2 = 3,
        activation_2 = act2
      )

    set.seed(1)
    fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

    orb_obj <- orbital(fit)
    preds <- predict(orb_obj, mtcars)
    exps <- predict(fit, mtcars)
    exps <- as.data.frame(exps)
    rownames(preds) <- NULL
    rownames(exps) <- NULL
    expect_equal(
      preds,
      exps,
      tolerance = 1e-5,
      info = paste0("activation_2 = ", act2)
    )
  }
})

# ---------- L7: brulee regression with extreme y_stats ----------

test_that("L7: brulee regression handles extreme y_stats (huge mean/tiny sd)", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("brulee")
  skip_if(!torch::torch_is_installed(), "torch not installed")

  data_huge <- mtcars
  data_huge$mpg <- mtcars$mpg + 1e6 # huge mean

  spec <- parsnip::mlp(hidden_units = 4, epochs = 20, engine = "brulee") |>
    parsnip::set_mode("regression")

  set.seed(1)
  fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, data_huge)

  orb_obj <- orbital(fit)
  preds <- predict(orb_obj, data_huge)
  exps <- predict(fit, data_huge)
  exps <- as.data.frame(exps)
  rownames(preds) <- NULL
  rownames(exps) <- NULL
  # Tolerance scaled to the predicted magnitude (~1e6).
  expect_equal(preds, exps, tolerance = 1e-3)

  # And tiny sd: scale y down by a huge factor so sd ~ 1e-4
  data_tiny <- mtcars
  data_tiny$mpg <- mtcars$mpg / 1e5

  set.seed(1)
  fit2 <- parsnip::fit(spec, mpg ~ disp + wt + hp, data_tiny)
  orb_obj2 <- orbital(fit2)
  preds2 <- predict(orb_obj2, data_tiny)
  exps2 <- predict(fit2, data_tiny)
  exps2 <- as.data.frame(exps2)
  rownames(preds2) <- NULL
  rownames(exps2) <- NULL
  expect_equal(preds2, exps2, tolerance = 1e-8)
})
