
# ── GroupNormalization groups=n_channels path (R audit rec #4) ───────────────

test_that("keras3 GroupNormalization groups=n_channels predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  # groups = n_channels = 8: each feature is its own group → output = beta_i
  x <- k$layers$GroupNormalization(groups = 8L)(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 5L, verbose = 0L)

  feature_names <- paste0("x", 1:4)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  expect_true(any(grepl("orbital_gn_", names(orb_obj))))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

# ── GroupNormalization unsupported groups error (audit rec #5) ───────────────

test_that("keras3 GroupNormalization general groups=2 predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  # groups=2 with 8 features: 8 %% 2 == 0, now supported
  x <- k$layers$GroupNormalization(groups = 2L)(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 5L, verbose = 0L)

  feature_names <- paste0("x", 1:4)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  expect_true(any(grepl("orbital_gn_gmean_", names(orb_obj))))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

test_that("keras3 GroupNormalization with indivisible groups raises cli error", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  # groups=5 with 8 features: 8 %% 5 != 0, must error
  x <- k$layers$GroupNormalization(groups = 5L)(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  model$fit(x_mat, rnorm(10), epochs = 1L, verbose = 0L)

  expect_error(
    orbital(
      model,
      mode = "regression",
      feature_names = paste0("x", 1:4)
    ),
    regexp = "GroupNormalization",
    ignore.case = TRUE
  )
})

test_that("keras3 RMSNormalization predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$RMSNormalization()(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 3L, verbose = 0L)

  feature_names <- paste0("x", 1:4)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  orb_obj <- orbital(model, mode = "regression", feature_names = feature_names)
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-4)
})

test_that("keras3 UnitNormalization predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$UnitNormalization()(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 3L, verbose = 0L)

  feature_names <- paste0("x", 1:4)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  orb_obj <- orbital(model, mode = "regression", feature_names = feature_names)
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-4)
})

test_that("keras3 UnitNormalization(axis=0) raises cli_abort", {
  .keras_skip()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(4L)(inp)
  x <- k$layers$UnitNormalization(axis = 0L)(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  feature_names <- paste0("x", 1:4)
  expect_error(
    orbital(model, mode = "regression", feature_names = feature_names),
    "axis"
  )
})
