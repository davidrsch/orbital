test_that("keras3 Sequential model with SELU activation predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  model <- k$Sequential(list(
    k$layers$Dense(6L, activation = "selu", input_shape = list(3L)),
    k$layers$Dense(1L)
  ))
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(30), nrow = 10, ncol = 3)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 5L, verbose = 0L)

  feature_names <- paste0("x", 1:3)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  hidden_exprs <- orb_obj[grepl("orbital_mlp", names(orb_obj))]
  # Every SELU expression must contain the SELU scale constant
  expect_true(any(grepl("1.0507009873554805", hidden_exprs)))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

test_that("keras3 Sequential model with Mish activation predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  model <- k$Sequential(list(
    k$layers$Dense(6L, activation = "mish", input_shape = list(3L)),
    k$layers$Dense(1L)
  ))
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(30), nrow = 10, ncol = 3)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 5L, verbose = 0L)

  feature_names <- paste0("x", 1:3)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  hidden_exprs <- orb_obj[grepl("orbital_mlp", names(orb_obj))]
  # Mish uses tanh(log(1 + exp(x))) — both tanh and log must appear
  expect_true(any(grepl("tanh", hidden_exprs)))
  expect_true(any(grepl("log", hidden_exprs)))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

test_that("keras3 Sequential model with Softplus activation predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  model <- k$Sequential(list(
    k$layers$Dense(6L, activation = "softplus", input_shape = list(3L)),
    k$layers$Dense(1L)
  ))
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(30), nrow = 10, ncol = 3)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 5L, verbose = 0L)

  feature_names <- paste0("x", 1:3)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  hidden_exprs <- orb_obj[grepl("orbital_mlp", names(orb_obj))]
  expect_true(any(grepl("log\\(1 \\+ exp", hidden_exprs)))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

test_that("keras3 Sequential model with Softsign activation predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  model <- k$Sequential(list(
    k$layers$Dense(6L, activation = "softsign", input_shape = list(3L)),
    k$layers$Dense(1L)
  ))
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(30), nrow = 10, ncol = 3)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 5L, verbose = 0L)

  feature_names <- paste0("x", 1:3)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  hidden_exprs <- orb_obj[grepl("orbital_mlp", names(orb_obj))]
  expect_true(any(grepl("abs", hidden_exprs)))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

# ── swish / hard_swish activation accuracy (audit rec #4) ────────────────────

test_that("keras3 Sequential model with swish activation predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  model <- k$Sequential(list(
    k$layers$Dense(6L, activation = "swish", input_shape = list(3L)),
    k$layers$Dense(1L)
  ))
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(30), nrow = 10, ncol = 3)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 5L, verbose = 0L)

  feature_names <- paste0("x", 1:3)
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

# ── multi-output model accuracy (audit rec #8) ───────────────────────────────

test_that("keras3 Functional two-output model predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  out1 <- k$layers$Dense(1L, name = "output_1")(x)
  out2 <- k$layers$Dense(1L, name = "output_2")(x)
  model <- k$Model(inputs = inp, outputs = list(out1, out2))
  model$compile(optimizer = "adam", loss = list("mse", "mse"))

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  y1 <- rnorm(10)
  y2 <- rnorm(10)
  model$fit(x_mat, list(y1, y2), epochs = 5L, verbose = 0L)

  feature_names <- paste0("x", 1:4)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  preds_orb <- predict(orb_obj, df)
  preds_keras <- model$predict(x_mat, verbose = 0L)
  # keras returns a list of two arrays for two outputs
  k_out1 <- as.numeric(preds_keras[[1L]])
  k_out2 <- as.numeric(preds_keras[[2L]])
  expect_equal(preds_orb$.pred_1, k_out1, tolerance = 1e-5)
  expect_equal(preds_orb$.pred_2, k_out2, tolerance = 1e-5)
})

# ── E2: coverage gap tests ────────────────────────────────────────────────────

test_that("keras3 ReLU(negative_slope=0.3) predictions match keras3 predict", {
  .keras_skip()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L)(inp)
  x <- k$layers$ReLU(negative_slope = 0.3)(x)
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
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

# ── multi-output Functional API: classification (R#32) ────────────────────────

test_that("keras3 Functional binary classifier predictions numerically match keras3 predict", {
  .keras_skip()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  out <- k$layers$Dense(1L, activation = "sigmoid")(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "binary_crossentropy")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  y_vec <- as.integer(rnorm(10) > 0)
  model$fit(x_mat, y_vec, epochs = 5L, verbose = 0L)

  feature_names <- paste0("x", 1:4)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  orb_obj <- orbital(
    model,
    mode = "classification",
    type = "prob",
    lvl = c("0", "1"),
    feature_names = feature_names
  )
  preds_orb <- predict(orb_obj, df)
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  # keras returns p(class=1); orbital .pred_1 should match
  expect_equal(preds_orb$.pred_1, preds_keras, tolerance = 1e-5)
  # probabilities sum to 1
  expect_equal(
    preds_orb$.pred_0 + preds_orb$.pred_1,
    rep(1, 10),
    tolerance = 1e-6
  )
})

# ---------------------------------------------------------------------------
# Error cases: unsupported and structurally-restricted activations
# ---------------------------------------------------------------------------

test_that("activation_expr raises for softmax as per-unit activation", {
  expect_error(
    activation_expr("softmax", "x"),
    regexp = "softmax"
  )
})

test_that("activation_expr raises for log_softmax as per-unit activation", {
  expect_error(
    activation_expr("log_softmax", "x"),
    regexp = "log_softmax"
  )
})

test_that("activation_expr raises for completely unknown activation name", {
  expect_error(
    activation_expr("not_a_real_activation", "x"),
    regexp = "not supported"
  )
})
