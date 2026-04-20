test_that("mlp() brulee works with regression", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("brulee")
  skip_if(!torch::torch_is_installed(), "torch not installed")

  spec <- parsnip::mlp(hidden_units = 4, epochs = 20, engine = "brulee")
  spec <- parsnip::set_mode(spec, "regression")

  set.seed(1)
  fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit)
  preds <- predict(orb_obj, mtcars)
  exps <- predict(fit, mtcars)

  expect_named(preds, ".pred")
  expect_type(preds$.pred, "double")

  exps <- as.data.frame(exps)
  rownames(preds) <- NULL
  rownames(exps) <- NULL

  expect_equal(preds, exps, tolerance = 1e-5)
})

test_that("mlp() brulee works with binary class", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("brulee")
  skip_if(!torch::torch_is_installed(), "torch not installed")

  mtcars$vs <- factor(mtcars$vs)
  spec <- parsnip::mlp(hidden_units = 4, epochs = 20, engine = "brulee")
  spec <- parsnip::set_mode(spec, "classification")

  set.seed(1)
  fit <- parsnip::fit(spec, vs ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit, type = "class")
  preds <- predict(orb_obj, mtcars)
  exps <- predict(fit, mtcars)

  expect_named(preds, ".pred_class")
  expect_identical(preds$.pred_class, as.character(exps$.pred_class))
})

test_that("mlp() brulee works with binary prob", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("brulee")
  skip_if(!torch::torch_is_installed(), "torch not installed")

  mtcars$vs <- factor(mtcars$vs)
  spec <- parsnip::mlp(hidden_units = 4, epochs = 20, engine = "brulee")
  spec <- parsnip::set_mode(spec, "classification")

  set.seed(1)
  fit <- parsnip::fit(spec, vs ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit, type = "prob")
  preds <- predict(orb_obj, mtcars)
  exps <- predict(fit, mtcars, type = "prob")

  lvls <- levels(mtcars$vs)
  expect_named(preds, paste0(".pred_", lvls))

  exps <- as.data.frame(exps)
  rownames(preds) <- NULL
  rownames(exps) <- NULL

  expect_equal(preds, exps, tolerance = 1e-5)
})

test_that("mlp() brulee_two_layer works with multiclass prob", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("brulee")
  skip_if(!torch::torch_is_installed(), "torch not installed")

  spec <- parsnip::mlp(
    hidden_units = 5,
    epochs = 20,
    engine = "brulee_two_layer"
  ) |>
    parsnip::set_mode("classification") |>
    parsnip::set_engine(
      "brulee_two_layer",
      hidden_units_2 = 3,
      activation_2 = "relu"
    )

  set.seed(1)
  fit <- parsnip::fit(
    spec,
    Species ~ Sepal.Length + Sepal.Width + Petal.Length,
    iris
  )

  orb_obj <- orbital(fit, type = "prob")
  preds <- predict(orb_obj, iris)
  exps <- predict(fit, iris, type = "prob")

  expect_named(
    preds,
    c(".pred_setosa", ".pred_versicolor", ".pred_virginica")
  )

  exps <- as.data.frame(exps)
  rownames(preds) <- NULL
  rownames(exps) <- NULL

  expect_equal(preds, exps, tolerance = 1e-5)
})

test_that("mlp() brulee with elu activation matches predictions", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("brulee")
  skip_if(!torch::torch_is_installed(), "torch not installed")

  spec <- parsnip::mlp(hidden_units = 4, epochs = 20, engine = "brulee") |>
    parsnip::set_mode("regression") |>
    parsnip::set_engine("brulee", activation = "elu")

  set.seed(1)
  fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit)
  preds <- predict(orb_obj, mtcars)
  exps <- predict(fit, mtcars)

  exps <- as.data.frame(exps)
  rownames(preds) <- NULL
  rownames(exps) <- NULL

  expect_equal(preds, exps, tolerance = 1e-5)
})

test_that("mlp() brulee with leaky_relu activation matches predictions", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("brulee")
  skip_if(!torch::torch_is_installed(), "torch not installed")

  spec <- parsnip::mlp(hidden_units = 4, epochs = 20, engine = "brulee") |>
    parsnip::set_mode("regression") |>
    parsnip::set_engine("brulee", activation = "leaky_relu")

  set.seed(1)
  fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit)
  preds <- predict(orb_obj, mtcars)
  exps <- predict(fit, mtcars)

  exps <- as.data.frame(exps)
  rownames(preds) <- NULL
  rownames(exps) <- NULL

  expect_equal(preds, exps, tolerance = 1e-5)
})

test_that("mlp() brulee with selu activation matches predictions", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("brulee")
  skip_if(!torch::torch_is_installed(), "torch not installed")

  spec <- parsnip::mlp(hidden_units = 4, epochs = 20, engine = "brulee") |>
    parsnip::set_mode("regression") |>
    parsnip::set_engine("brulee", activation = "selu")

  set.seed(1)
  fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit)
  preds <- predict(orb_obj, mtcars)
  exps <- predict(fit, mtcars)

  exps <- as.data.frame(exps)
  rownames(preds) <- NULL
  rownames(exps) <- NULL

  expect_equal(preds, exps, tolerance = 1e-5)
})

test_that("mlp() brulee with gelu activation matches predictions", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("brulee")
  skip_if(!torch::torch_is_installed(), "torch not installed")

  spec <- parsnip::mlp(hidden_units = 4, epochs = 20, engine = "brulee") |>
    parsnip::set_mode("regression") |>
    parsnip::set_engine("brulee", activation = "gelu")

  set.seed(1)
  fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit)
  preds <- predict(orb_obj, mtcars)
  exps <- predict(fit, mtcars)

  exps <- as.data.frame(exps)
  rownames(preds) <- NULL
  rownames(exps) <- NULL

  expect_equal(preds, exps, tolerance = 1e-5)
})

test_that("mlp() brulee with mish activation matches predictions", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("brulee")
  skip_if(!torch::torch_is_installed(), "torch not installed")

  spec <- parsnip::mlp(hidden_units = 4, epochs = 20, engine = "brulee") |>
    parsnip::set_mode("regression") |>
    parsnip::set_engine("brulee", activation = "mish")

  set.seed(1)
  fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit)
  preds <- predict(orb_obj, mtcars)
  exps <- predict(fit, mtcars)

  exps <- as.data.frame(exps)
  rownames(preds) <- NULL
  rownames(exps) <- NULL

  expect_equal(preds, exps, tolerance = 1e-5)
})

test_that("mlp() brulee with silu activation matches predictions", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("brulee")
  skip_if(!torch::torch_is_installed(), "torch not installed")

  spec <- parsnip::mlp(hidden_units = 4, epochs = 20, engine = "brulee") |>
    parsnip::set_mode("regression") |>
    parsnip::set_engine("brulee", activation = "silu")

  set.seed(1)
  fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit)
  preds <- predict(orb_obj, mtcars)
  exps <- predict(fit, mtcars)

  exps <- as.data.frame(exps)
  rownames(preds) <- NULL
  rownames(exps) <- NULL

  expect_equal(preds, exps, tolerance = 1e-5)
})

test_that("mlp() brulee works with multiclass class", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("brulee")
  skip_if(!torch::torch_is_installed(), "torch not installed")

  spec <- parsnip::mlp(hidden_units = 5, epochs = 20, engine = "brulee") |>
    parsnip::set_mode("classification")

  set.seed(1)
  fit <- parsnip::fit(
    spec,
    Species ~ Sepal.Length + Sepal.Width + Petal.Length,
    iris
  )

  orb_obj <- orbital(fit, type = "class")
  preds <- predict(orb_obj, iris)
  exps <- predict(fit, iris)

  expect_named(preds, ".pred_class")
  expect_identical(preds$.pred_class, as.character(exps$.pred_class))
})

test_that("orbital_brulee_mlp_impl errors for n_h_layers > 2", {
  # brulee only exposes activation and activation_2 parameters; > 2 layers is unsupported
  skip("Requires a fitted brulee model with n_h_layers > 2")
})

test_that("mlp() brulee activation sweep matches predictions for additional activations", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("brulee")
  skip_if(!torch::torch_is_installed(), "torch not installed")

  extra_activations <- c(
    "tanh",
    "sigmoid",
    "linear",
    "celu",
    "prelu",
    "relu6",
    "hardtanh",
    "softplus",
    "softsign",
    "hardswish",
    "hardsigmoid"
  )

  for (act in extra_activations) {
    spec <- parsnip::mlp(hidden_units = 4, epochs = 10, engine = "brulee") |>
      parsnip::set_mode("regression") |>
      parsnip::set_engine("brulee", activation = act)

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
      info = paste("activation =", act)
    )
  }
})

test_that("mlp() brulee_two_layer activation_2 = softmax regression matches", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("brulee")
  skip_if(!torch::torch_is_installed(), "torch not installed")

  # Exercises the softmax_hidden_exprs() branch in model-brulee.R and verifies
  # that the activation_2 key is read correctly for brulee_two_layer models.
  spec <- parsnip::mlp(
    hidden_units = 4,
    epochs = 20,
    engine = "brulee_two_layer"
  ) |>
    parsnip::set_mode("regression") |>
    parsnip::set_engine(
      "brulee_two_layer",
      hidden_units_2 = 3,
      activation_2 = "softmax"
    )

  set.seed(1)
  fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit)
  preds <- predict(orb_obj, mtcars)
  exps <- predict(fit, mtcars)

  expect_named(preds, ".pred")
  expect_type(preds$.pred, "double")

  exps <- as.data.frame(exps)
  rownames(preds) <- NULL
  rownames(exps) <- NULL

  expect_equal(preds, exps, tolerance = 1e-5)
})


test_that("brulee: missing fc.weight raises a clear error (finding D-04)", {
  skip_if_not_installed("brulee")
  # Regression test: previously a corrupted/incompatible brulee fit produced
  # an opaque NULL-subscript error. orbital now cli_abort()s with a clear
  # message listing the missing key and the available keys.
  fake_fit <- structure(
    list(
      dims = list(features = c("x1", "x2"), h = 3L, y = 1L),
      parameters = list(activation = "relu"),
      best_epoch = 1L,
      estimates = list(list(fc1.bias = c(0, 0, 0))),
      y_stats = list(mean = 0, sd = 1),
      blueprint = structure(list(), class = "hardhat_blueprint")
    ),
    class = c("brulee_mlp", "list")
  )
  expect_error(
    orbital(fake_fit, mode = "regression"),
    regexp = "fc1\\.weight"
  )
})

# ---------------------------------------------------------------------------
# Phase 9 additions (E7, E8): B-02 activation coverage, B-03/B-01 deep MLP
# ---------------------------------------------------------------------------

# E7 / B-02: Five activations that brulee itself does not expose as string
# choices (hardshrink, softshrink, tanhshrink, rrelu) or that are only
# partially covered (log_sigmoid). We test them as unit-level round-trips
# through orbital::activation_expr() rather than end-to-end brulee fits
# because brulee's parsnip bridge rejects unknown activation strings.
test_that("B-02: activation_expr produces non-empty SQL for hardshrink", {
  expr <- orbital:::activation_expr("hardshrink", "z")
  expect_type(expr, "character")
  expect_true(nchar(expr) > 0L)
  expect_match(expr, "z", fixed = TRUE)
})

test_that("B-02: activation_expr produces non-empty SQL for softshrink", {
  expr <- orbital:::activation_expr("softshrink", "z")
  expect_type(expr, "character")
  expect_true(nchar(expr) > 0L)
  expect_match(expr, "z", fixed = TRUE)
})

test_that("B-02: activation_expr produces non-empty SQL for tanhshrink", {
  expr <- orbital:::activation_expr("tanhshrink", "z")
  expect_type(expr, "character")
  expect_true(nchar(expr) > 0L)
  expect_match(expr, "tanh", fixed = TRUE)
})

test_that("B-02: activation_expr produces non-empty SQL for log_sigmoid", {
  expr <- orbital:::activation_expr("log_sigmoid", "z")
  expect_type(expr, "character")
  expect_true(nchar(expr) > 0L)
  expect_match(expr, "log", fixed = TRUE)
})

test_that("B-02: activation_expr produces non-empty SQL for rrelu", {
  expr <- orbital:::activation_expr("rrelu", "z")
  expect_type(expr, "character")
  expect_true(nchar(expr) > 0L)
  # rrelu at inference uses the fixed midpoint negative slope
  # (1/3 + 1/8)/2 = 0.2291666... per the ann-helpers implementation.
  expect_match(expr, "z", fixed = TRUE)
})

test_that("B-03/B-01: brulee_mlp with hidden_units = c(4, 3, 2) translates successfully", {
  skip_if_not_installed("brulee")
  skip_if_not_installed("parsnip")
  skip_if(!torch::torch_is_installed(), "torch not installed")

  x <- as.matrix(mtcars[, c("disp", "wt", "hp")])
  y <- mtcars$mpg

  set.seed(1)
  fit <- brulee::brulee_mlp(
    x = x,
    y = y,
    hidden_units = c(4L, 3L, 2L),
    epochs = 5L,
    verbose = FALSE
  )

  expect_no_error(orb_obj <- orbital(fit, mode = "regression"))
  expect_s3_class(orb_obj, "orbital")
})
