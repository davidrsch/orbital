# Integration tests for keras3 activation-specific Sequential/Functional models
# These tests are skipped when required packages are not installed.

# ── activations: ELU, GELU, hard_swish ───────────────────────────────────────

test_that("keras3 Sequential model with ELU activation translates correctly", {
  skip_if_no_keras3()

  k <- reticulate::import("keras")
  model <- k$Sequential(list(
    k$layers$Dense(4L, activation = "elu", input_shape = list(3L)),
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
  expect_true(is.character(orb_obj))
  hidden_exprs <- orb_obj[grepl("orbital_mlp", names(orb_obj))]
  # ELU expression must contain exp()
  expect_true(any(grepl("exp\\(", hidden_exprs)))
})

test_that("keras3 Sequential model with GELU activation translates correctly", {
  skip_if_no_keras3()

  k <- reticulate::import("keras")
  model <- k$Sequential(list(
    k$layers$Dense(4L, activation = "gelu", input_shape = list(3L)),
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
  expect_true(is.character(orb_obj))
  hidden_exprs <- orb_obj[grepl("orbital_mlp", names(orb_obj))]
  # After the GELU fix the expression uses the A&S erf approximation
  # (exact form); the 1/sqrt(2) constant is the distinguishing token.
  expect_true(any(grepl("0.7071067811865476", hidden_exprs)))
})

test_that("keras3 Sequential model with hard_sigmoid activation translates correctly", {
  skip_if_no_keras3()

  k <- reticulate::import("keras")
  model <- k$Sequential(list(
    k$layers$Dense(4L, activation = "hard_sigmoid", input_shape = list(3L)),
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
  expect_true(is.character(orb_obj))
  hidden_exprs <- orb_obj[grepl("orbital_mlp", names(orb_obj))]
  expect_true(any(grepl("0\\.2", hidden_exprs)))
})

# ── multiclass probability output (R #19) ────────────────────────────────────

test_that("keras3 Sequential multiclass model outputs class probabilities", {
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
    type = "prob",
    lvl = c("cat_a", "cat_b", "cat_c"),
    feature_names = paste0("x", 1:4)
  )
  expect_true(is.character(orb_obj))
  expect_true(".pred_cat_a" %in% names(orb_obj))
  expect_true(".pred_cat_b" %in% names(orb_obj))
  expect_true(".pred_cat_c" %in% names(orb_obj))
})

test_that("keras3 PReLU predictions match keras3 predict (regression)", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(3L))
  x <- k$layers$Dense(6L)(inp)
  x <- k$layers$PReLU()(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
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

test_that("keras3 Concatenate predictions match keras3 predict (regression)", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  branch_a <- k$layers$Dense(3L, activation = "relu")(inp)
  branch_b <- k$layers$Dense(3L, activation = "tanh")(inp)
  merged <- k$layers$Concatenate()(list(branch_a, branch_b))
  out <- k$layers$Dense(1L)(merged)
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

test_that("keras3 ELU activation predictions match keras3 predict (regression)", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  model <- k$Sequential(list(
    k$layers$Dense(6L, activation = "elu", input_shape = list(3L)),
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

test_that("keras3 GELU activation predictions match keras3 predict (regression)", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  model <- k$Sequential(list(
    k$layers$Dense(6L, activation = "gelu", input_shape = list(3L)),
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

# ── activations: selu, mish, softplus, softsign, leaky_relu (R audit rec #1) ─
