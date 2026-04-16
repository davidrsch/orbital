# Integration tests: structural layers (Add, Dropout, Flatten, Reshape, multi-output classification, SpatialDropout, GaussianDropout, AlphaDropout).
# These tests are skipped when required packages are not installed.

test_that("keras3 Functional model with Add (residual) predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(4L, activation = "relu")(inp)
  out_t <- k$layers$Add()(list(x, inp))
  out <- k$layers$Dense(1L)(out_t)
  model <- k$Model(inp, out)
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
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})


test_that("keras3 model with Dropout layer predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  x <- k$layers$Dropout(rate = 0.5)(x)
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
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})


test_that("keras3 model with Flatten layer predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$Flatten()(x)
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
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})


test_that("keras3 model with Reshape layer predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$Reshape(list(6L))(x)
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
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})


test_that("keras3 Functional multi-output: two classification heads produce named prediction columns", {
  .keras_skip()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  out1 <- k$layers$Dense(1L, name = "head_1")(x)
  out2 <- k$layers$Dense(1L, name = "head_2")(x)
  model <- k$Model(inputs = inp, outputs = list(out1, out2))
  model$compile(
    optimizer = "adam",
    loss = list("binary_crossentropy", "binary_crossentropy")
  )

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  y1 <- as.integer(rnorm(10) > 0)
  y2 <- as.integer(rnorm(10) > 0)
  model$fit(x_mat, list(y1, y2), epochs = 3L, verbose = 0L)

  feature_names <- paste0("x", 1:4)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  # Two output heads → orbital treats as 2-class distribution
  orb_obj <- orbital(
    model,
    mode = "classification",
    type = "prob",
    lvl = c("head_1", "head_2"),
    feature_names = feature_names
  )
  preds_orb <- predict(orb_obj, df)
  expect_true(".pred_head_1" %in% names(preds_orb))
  expect_true(".pred_head_2" %in% names(preds_orb))
  # Probabilities must be non-negative and sum to 1
  expect_true(all(preds_orb$.pred_head_1 >= 0))
  expect_true(all(preds_orb$.pred_head_2 >= 0))
  expect_equal(
    preds_orb$.pred_head_1 + preds_orb$.pred_head_2,
    rep(1, 10),
    tolerance = 1e-6
  )
})

# ── Subtract merge layer ──────────────────────────────────────────────────────

test_that("keras3 model with SpatialDropout1D layer predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(6L, 4L))
  x <- k$layers$Conv1D(
    8L,
    kernel_size = 3L,
    activation = "relu",
    padding = "same"
  )(inp)
  x <- k$layers$SpatialDropout1D(rate = 0.3)(x)
  x <- k$layers$GlobalAveragePooling1D()(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- array(rnorm(240), dim = c(10L, 6L, 4L))
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 5L, verbose = 0L)

  x_flat <- matrix(as.numeric(x_mat), nrow = 10L)
  df <- as.data.frame(x_flat)
  names(df) <- paste0("x", 1:24)
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = paste0("x", 1:24)
  )
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})


test_that("keras3 model with GaussianDropout layer predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  x <- k$layers$GaussianDropout(rate = 0.3)(x)
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
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})


test_that("keras3 model with AlphaDropout layer predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "selu")(inp)
  x <- k$layers$AlphaDropout(rate = 0.3)(x)
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
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

# ── Dot (normalize=TRUE) ──────────────────────────────────────────────────────
