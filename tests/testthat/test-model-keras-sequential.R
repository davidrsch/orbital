# Integration tests for keras3 / kerasnip Sequential MLP models
# Activation tests -> test-model-keras-sequential-activations.R
# EinsumDense tests -> test-model-keras-einsumdense.R

#  keras3 (modern API, Sequential)

test_that("mlp() keras3 Sequential works with regression", {
  skip_if_no_keras3()

  # Build a simple keras3 Sequential MLP directly (not via parsnip)
  k <- reticulate::import("keras")
  model <- k$Sequential(list(
    k$layers$Dense(4L, activation = "relu", input_shape = list(3L)),
    k$layers$Dense(1L)
  ))
  model$compile(optimizer = "adam", loss = "mse")

  # Fit on tiny data
  set.seed(42)
  x_mat <- matrix(rnorm(30), nrow = 10, ncol = 3)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 2L, verbose = 0L)

  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = paste0("x", 1:3)
  )
  expect_true(is.character(orb_obj))
  expect_named(orb_obj, ".pred", ignore.order = TRUE)
})

test_that("mlp() keras3 Sequential works with binary classification", {
  skip_if_no_keras3()

  k <- reticulate::import("keras")
  model <- k$Sequential(list(
    k$layers$Dense(4L, activation = "relu", input_shape = list(3L)),
    k$layers$Dense(1L, activation = "sigmoid")
  ))
  model$compile(optimizer = "adam", loss = "binary_crossentropy")

  set.seed(42)
  x_mat <- matrix(rnorm(30), nrow = 10, ncol = 3)
  y_vec <- as.integer(rnorm(10) > 0)
  model$fit(x_mat, y_vec, epochs = 2L, verbose = 0L)

  orb_obj <- orbital(
    model,
    mode = "classification",
    type = "class",
    lvl = c("0", "1"),
    feature_names = paste0("x", 1:3)
  )
  expect_true(is.character(orb_obj))
  expect_true(".pred_class" %in% names(orb_obj))
})

test_that("mlp() keras3 Sequential works with multiclass (softmax)", {
  skip_if_no_keras3()

  k <- reticulate::import("keras")
  model <- k$Sequential(list(
    k$layers$Dense(8L, activation = "relu", input_shape = list(4L)),
    k$layers$Dense(3L, activation = "softmax")
  ))
  model$compile(optimizer = "adam", loss = "sparse_categorical_crossentropy")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  y_vec <- sample(0:2, 10, replace = TRUE)
  model$fit(x_mat, y_vec, epochs = 2L, verbose = 0L)

  orb_obj <- orbital(
    model,
    mode = "classification",
    type = "class",
    lvl = c("0", "1", "2"),
    feature_names = paste0("x", 1:4)
  )
  expect_true(is.character(orb_obj))
})

# ── swish and mish activations ────────────────────────────────────────────────

test_that("keras3 model with swish activation is translatable", {
  skip_if_no_keras3()

  k <- reticulate::import("keras")
  model <- k$Sequential(list(
    k$layers$Dense(4L, activation = "swish", input_shape = list(3L)),
    k$layers$Dense(1L)
  ))
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(30), nrow = 10, ncol = 3)
  model$fit(x_mat, rnorm(10), epochs = 2L, verbose = 0L)

  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = paste0("x", 1:3)
  )
  # swish expressions should contain the characteristic "(1 / (1 + exp(-" pattern
  hidden_exprs <- orb_obj[grepl("orbital_mlp", names(orb_obj))]
  expect_true(any(grepl("exp", hidden_exprs)))
})

# ── kerasnip (skip if not installed) ─────────────────────────────────────────

test_that("mlp() kerasnip engine works with regression", {
  skip_if_not_installed("kerasnip")
  skip_if_not_installed("parsnip")
  skip_if_not(
    tryCatch(
      {
        parsnip::mlp(engine = "kerasnip")
        TRUE
      },
      error = function(e) FALSE
    ),
    "kerasnip engine not registered with parsnip"
  )

  spec <- parsnip::mlp(hidden_units = 3, epochs = 5, engine = "kerasnip")
  spec <- parsnip::set_mode(spec, "regression")

  set.seed(1)
  fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit)
  preds <- predict(orb_obj, mtcars)

  expect_named(preds, ".pred")
  expect_type(preds$.pred, "double")
})

test_that("mlp() kerasnip engine works with binary classification", {
  skip_if_not_installed("kerasnip")
  skip_if_not_installed("parsnip")
  skip_if_not(
    tryCatch(
      {
        parsnip::mlp(engine = "kerasnip")
        TRUE
      },
      error = function(e) FALSE
    ),
    "kerasnip engine not registered with parsnip"
  )

  mtcars$vs <- factor(mtcars$vs)
  spec <- parsnip::mlp(hidden_units = 3, epochs = 5, engine = "kerasnip")
  spec <- parsnip::set_mode(spec, "classification")

  set.seed(1)
  fit <- parsnip::fit(spec, vs ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit, type = "class")
  preds <- predict(orb_obj, mtcars)

  expect_named(preds, ".pred_class")
  expect_type(preds$.pred_class, "character")
})
