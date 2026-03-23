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
