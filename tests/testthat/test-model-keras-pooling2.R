# Integration tests for keras3 pooling: ZeroPadding1D, Cropping1D, padding=same, UpSampling1D, Adaptive variants.
# These tests are skipped when required packages are not installed.

test_that("keras3 ZeroPadding1D predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  T_len <- 3L
  C_in <- 2L
  inp <- k$Input(shape = list(T_len, C_in))
  x <- k$layers$ZeroPadding1D(padding = 1L)(inp)
  x <- k$layers$Flatten()(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  n_row <- 10L
  x_flat <- matrix(
    rnorm(n_row * T_len * C_in),
    nrow = n_row,
    ncol = T_len * C_in
  )
  y_vec <- rnorm(n_row)
  x_3d <- array(x_flat, dim = c(n_row, T_len, C_in))
  model$fit(x_3d, y_vec, epochs = 3L, verbose = 0L)

  feature_names <- paste0("x", seq_len(T_len * C_in))
  df <- as.data.frame(x_flat)
  names(df) <- feature_names
  orb_obj <- orbital(model, mode = "regression", feature_names = feature_names)
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_3d, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-4)
})

# ── Cropping1D and RepeatVector ───────────────────────────────────────────────

test_that("keras3 Cropping1D predictions match keras3 predict", {
  .keras_skip()
  k <- reticulate::import("keras")
  T_len <- 6L
  C_in <- 2L
  inp <- k$Input(shape = list(T_len, C_in))
  x <- k$layers$Cropping1D(cropping = list(1L, 1L))(inp) # removes 1 step each end
  x <- k$layers$Flatten()(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  n_row <- 10L
  x_flat <- matrix(
    rnorm(n_row * T_len * C_in),
    nrow = n_row,
    ncol = T_len * C_in
  )
  x_3d <- array(x_flat, dim = c(n_row, T_len, C_in))
  model$fit(x_3d, rnorm(n_row), epochs = 2L, verbose = 0L)

  feature_names <- paste0("x", seq_len(T_len * C_in))
  df <- as.data.frame(x_flat)
  names(df) <- feature_names
  orb_obj <- orbital(model, mode = "regression", feature_names = feature_names)
  expect_equal(
    predict(orb_obj, df)$.pred,
    as.numeric(model$predict(x_3d, verbose = 0L)),
    tolerance = 1e-4
  )
})

test_that("keras3 AveragePooling1D(padding='same') predictions match keras3 predict", {
  .keras_skip()
  k <- reticulate::import("keras")
  T_len <- 5L
  C_in <- 2L
  inp <- k$Input(shape = list(T_len, C_in))
  x <- k$layers$AveragePooling1D(
    pool_size = 3L,
    strides = 2L,
    padding = "same"
  )(inp)
  x <- k$layers$Flatten()(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  n_row <- 10L
  x_flat <- matrix(
    rnorm(n_row * T_len * C_in),
    nrow = n_row,
    ncol = T_len * C_in
  )
  x_3d <- array(x_flat, dim = c(n_row, T_len, C_in))
  model$fit(x_3d, rnorm(n_row), epochs = 3L, verbose = 0L)

  feature_names <- paste0("x", seq_len(T_len * C_in))
  df <- as.data.frame(x_flat)
  names(df) <- feature_names
  orb_obj <- orbital(model, mode = "regression", feature_names = feature_names)
  expect_equal(
    predict(orb_obj, df)$.pred,
    as.numeric(model$predict(x_3d, verbose = 0L)),
    tolerance = 1e-4
  )
})

# ── UpSampling1D layer ────────────────────────────────────────────────────────

test_that("keras3 UpSampling1D predictions match keras3 predict", {
  .keras_skip()
  k <- reticulate::import("keras")
  T_len <- 3L
  C_in <- 2L
  inp <- k$Input(shape = list(T_len, C_in))
  x <- k$layers$UpSampling1D(size = 2L)(inp)
  x <- k$layers$Flatten()(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_arr <- array(rnorm(10 * T_len * C_in), dim = c(10L, T_len, C_in))
  model$fit(x_arr, rnorm(10), epochs = 2L, verbose = 0L)

  x_flat <- matrix(x_arr, nrow = 10, ncol = T_len * C_in)
  df <- as.data.frame(x_flat)
  names(df) <- paste0("x", seq_len(T_len * C_in))
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = names(df)
  )
  expect_equal(
    predict(orb_obj, df)$.pred,
    as.numeric(model$predict(x_arr, verbose = 0L)),
    tolerance = 1e-4
  )
})

# ── AdaptiveAveragePooling1D ──────────────────────────────────────────────────

test_that("keras3 AdaptiveAveragePooling1D predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(6L, 4L))
  x <- k$layers$Conv1D(
    8L,
    kernel_size = 3L,
    padding = "same",
    activation = "relu"
  )(inp)
  x <- k$layers$AdaptiveAveragePooling1D(output_size = 3L)(x)
  x <- k$layers$Flatten()(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")
  set.seed(42)
  x_mat <- array(rnorm(240), dim = c(10L, 6L, 4L))
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 3L, verbose = 0L)
  x_flat <- matrix(as.numeric(x_mat), nrow = 10L)
  feature_names <- paste0("x", seq_len(ncol(x_flat)))
  df <- as.data.frame(x_flat)
  names(df) <- feature_names
  orb_obj <- orbital(model, mode = "regression", feature_names = feature_names)
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

# ── AdaptiveMaxPooling1D ──────────────────────────────────────────────────────

test_that("keras3 AdaptiveMaxPooling1D predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(6L, 4L))
  x <- k$layers$Conv1D(
    8L,
    kernel_size = 3L,
    padding = "same",
    activation = "relu"
  )(inp)
  x <- k$layers$AdaptiveMaxPooling1D(output_size = 3L)(x)
  x <- k$layers$Flatten()(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")
  set.seed(42)
  x_mat <- array(rnorm(240), dim = c(10L, 6L, 4L))
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 3L, verbose = 0L)
  x_flat <- matrix(as.numeric(x_mat), nrow = 10L)
  feature_names <- paste0("x", seq_len(ncol(x_flat)))
  df <- as.data.frame(x_flat)
  names(df) <- feature_names
  orb_obj <- orbital(model, mode = "regression", feature_names = feature_names)
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})
