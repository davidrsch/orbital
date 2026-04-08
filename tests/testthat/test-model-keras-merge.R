# Integration tests for keras3 merge layer handlers
# These tests are skipped when required packages are not installed.


test_that("mlp() keras3 Functional model with Add (residual) works", {
  skip_if_no_keras3()

  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(4L, activation = "relu")(inp)
  out_t <- k$layers$Add()(list(x, inp)) # residual: 4 → 4 + input
  out <- k$layers$Dense(1L)(out_t)
  model <- k$Model(inp, out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 2L, verbose = 0L)

  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = paste0("x", 1:4)
  )
  expect_true(is.character(orb_obj))
  expect_named(orb_obj, ".pred", ignore.order = TRUE)
})


test_that("keras3 Multiply merge layer predictions match keras3 predict", {
  .keras_skip()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  a <- k$layers$Dense(4L)(inp)
  b <- k$layers$Dense(4L)(inp)
  x <- k$layers$Multiply()(list(a, b))
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  model$fit(x_mat, rnorm(10), epochs = 2L, verbose = 0L)

  df <- as.data.frame(x_mat)
  names(df) <- paste0("x", 1:4)
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = paste0("x", 1:4)
  )
  expect_equal(
    predict(orb_obj, df)$.pred,
    as.numeric(model$predict(x_mat, verbose = 0L)),
    tolerance = 1e-4
  )
})


test_that("keras3 Average merge layer predictions match keras3 predict", {
  .keras_skip()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  a <- k$layers$Dense(4L)(inp)
  b <- k$layers$Dense(4L)(inp)
  x <- k$layers$Average()(list(a, b))
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  model$fit(x_mat, rnorm(10), epochs = 2L, verbose = 0L)

  df <- as.data.frame(x_mat)
  names(df) <- paste0("x", 1:4)
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = paste0("x", 1:4)
  )
  expect_equal(
    predict(orb_obj, df)$.pred,
    as.numeric(model$predict(x_mat, verbose = 0L)),
    tolerance = 1e-4
  )
})


test_that("keras3 Maximum merge layer predictions match keras3 predict", {
  .keras_skip()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  a <- k$layers$Dense(4L)(inp)
  b <- k$layers$Dense(4L)(inp)
  x <- k$layers$Maximum()(list(a, b))
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  model$fit(x_mat, rnorm(10), epochs = 2L, verbose = 0L)

  df <- as.data.frame(x_mat)
  names(df) <- paste0("x", 1:4)
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = paste0("x", 1:4)
  )
  expect_equal(
    predict(orb_obj, df)$.pred,
    as.numeric(model$predict(x_mat, verbose = 0L)),
    tolerance = 1e-4
  )
})


test_that("keras3 Minimum merge layer predictions match keras3 predict", {
  .keras_skip()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  a <- k$layers$Dense(4L)(inp)
  b <- k$layers$Dense(4L)(inp)
  x <- k$layers$Minimum()(list(a, b))
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  model$fit(x_mat, rnorm(10), epochs = 2L, verbose = 0L)

  df <- as.data.frame(x_mat)
  names(df) <- paste0("x", 1:4)
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = paste0("x", 1:4)
  )
  expect_equal(
    predict(orb_obj, df)$.pred,
    as.numeric(model$predict(x_mat, verbose = 0L)),
    tolerance = 1e-4
  )
})


test_that("keras3 Dot merge layer predictions match keras3 predict", {
  .keras_skip()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  a <- k$layers$Dense(4L)(inp)
  b <- k$layers$Dense(4L)(inp)
  # axes = -1L means dot product over last axis
  x <- k$layers$Dot(axes = -1L)(list(a, b))
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  model$fit(x_mat, rnorm(10), epochs = 2L, verbose = 0L)

  df <- as.data.frame(x_mat)
  names(df) <- paste0("x", 1:4)
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = paste0("x", 1:4)
  )
  expect_equal(
    predict(orb_obj, df)$.pred,
    as.numeric(model$predict(x_mat, verbose = 0L)),
    tolerance = 1e-4
  )
})

# ── Permute layer ─────────────────────────────────────────────────────────────


test_that("keras3 Subtract merge layer predictions match keras3 predict", {
  .keras_skip()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  a <- k$layers$Dense(4L)(inp)
  b <- k$layers$Dense(4L)(inp)
  x <- k$layers$Subtract()(list(a, b))
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  model$fit(x_mat, rnorm(10), epochs = 2L, verbose = 0L)

  df <- as.data.frame(x_mat)
  names(df) <- paste0("x", 1:4)
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = paste0("x", 1:4)
  )
  expect_equal(
    predict(orb_obj, df)$.pred,
    as.numeric(model$predict(x_mat, verbose = 0L)),
    tolerance = 1e-4
  )
})


test_that("keras3 Dot(normalize=TRUE) cosine similarity predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  a <- k$layers$Dense(4L)(inp)
  b <- k$layers$Dense(4L)(inp)
  x <- k$layers$Dot(axes = -1L, normalize = TRUE)(list(a, b))
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  model$fit(x_mat, rnorm(10), epochs = 2L, verbose = 0L)

  df <- as.data.frame(x_mat)
  names(df) <- paste0("x", 1:4)
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = paste0("x", 1:4)
  )
  expect_equal(
    predict(orb_obj, df)$.pred,
    as.numeric(model$predict(x_mat, verbose = 0L)),
    tolerance = 1e-4
  )
})

# ── EinsumDense layer tests (TEST-2) ─────────────────────────────────────────

