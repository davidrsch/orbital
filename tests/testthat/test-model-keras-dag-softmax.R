# Integration tests for keras3 Functional models: multi-output + Softmax + ELU.
# These tests are skipped when required packages are not installed.

test_that("keras3 Functional model with two output Dense layers produces named predictions", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  out1 <- k$layers$Dense(1L, name = "output_1")(x)
  out2 <- k$layers$Dense(1L, name = "output_2")(x)
  model <- k$Model(inputs = inp, outputs = list(out1, out2))
  model$compile(
    optimizer = "adam",
    loss = list("mse", "mse")
  )

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  y1 <- rnorm(10)
  y2 <- rnorm(10)
  model$fit(x_mat, list(y1, y2), epochs = 3L, verbose = 0L)

  feature_names <- paste0("x", 1:4)
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  expect_true(is.character(orb_obj))
  # Both output expressions should be present
  expect_true(".pred_1" %in% names(orb_obj))
  expect_true(".pred_2" %in% names(orb_obj))
})


test_that("keras3 multi-output Functional model predictions numerically match keras3 predict", {
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
  model$fit(x_mat, list(y1, y2), epochs = 3L, verbose = 0L)

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
  expect_equal(
    preds_orb$.pred_1,
    as.numeric(preds_keras[[1L]]),
    tolerance = 1e-5
  )
  expect_equal(
    preds_orb$.pred_2,
    as.numeric(preds_keras[[2L]]),
    tolerance = 1e-5
  )
})


test_that("keras3 Functional model with hard_swish Activation layer predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(3L))
  x <- k$layers$Dense(6L)(inp)
  x <- k$layers$Activation("hard_swish")(x)
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

# ── standalone Softmax layer accuracy (audit rec #6) ─────────────────────────

test_that("keras3 Functional model with standalone Softmax layer predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  out <- k$layers$Dense(3L)(x)
  out <- k$layers$Softmax()(out)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "sparse_categorical_crossentropy")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  y_vec <- sample(0:2, 10, replace = TRUE)
  model$fit(x_mat, y_vec, epochs = 3L, verbose = 0L)

  feature_names <- paste0("x", 1:4)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  orb_obj <- orbital(
    model,
    mode = "classification",
    type = "prob",
    lvl = c("0", "1", "2"),
    feature_names = feature_names
  )
  expect_true(any(grepl("orbital_sm_", names(orb_obj))))
  preds_orb <- predict(orb_obj, df)
  preds_keras <- model$predict(x_mat, verbose = 0L)
  # Probabilities from orbital must match keras to within 1e-5
  expect_equal(
    preds_orb$.pred_0,
    as.numeric(preds_keras[, 1L]),
    tolerance = 1e-5
  )
  expect_equal(
    preds_orb$.pred_1,
    as.numeric(preds_keras[, 2L]),
    tolerance = 1e-5
  )
  expect_equal(
    preds_orb$.pred_2,
    as.numeric(preds_keras[, 3L]),
    tolerance = 1e-5
  )
})

# ── standalone Activation(log_softmax) DAG path (audit rec #7) ───────────────

test_that("keras3 Functional model with standalone Activation(log_softmax) DAG path", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(3L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  out <- k$layers$Dense(3L)(x)
  out <- k$layers$Activation("log_softmax")(out)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "sparse_categorical_crossentropy")

  set.seed(42)
  x_mat <- matrix(rnorm(30), nrow = 10, ncol = 3)
  y_vec <- sample(0:2, 10, replace = TRUE)
  model$fit(x_mat, y_vec, epochs = 3L, verbose = 0L)

  feature_names <- paste0("x", 1:3)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  # The log_softmax activation produces orbital_act_ intermediate expressions
  expect_true(any(grepl("orbital_act_", names(orb_obj))))
  # All log_softmax outputs should contain a log subtraction pattern
  act_exprs <- orb_obj[grepl("orbital_act_", names(orb_obj))]
  expect_true(any(grepl("log", act_exprs)))
  # Numerical accuracy: log_softmax == log(softmax)
  preds_orb <- predict(orb_obj, df)
  preds_keras <- model$predict(x_mat, verbose = 0L)
  # orbital outputs log-probabilities; exp of orbital must equal keras probs
  expect_equal(
    exp(as.numeric(preds_orb[[1L]])),
    as.numeric(preds_keras[, 1L]),
    tolerance = 1e-5
  )
  expect_equal(
    exp(as.numeric(preds_orb[[2L]])),
    as.numeric(preds_keras[, 2L]),
    tolerance = 1e-5
  )
  expect_equal(
    exp(as.numeric(preds_orb[[3L]])),
    as.numeric(preds_keras[, 3L]),
    tolerance = 1e-5
  )
})

# ── standalone ELU layer (R audit fix #2) ────────────────────────────────────

test_that("keras3 Functional model with ELU layer predictions match", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L)(inp)
  x <- k$layers$ELU(alpha = 0.5)(x)
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
  # ELU layer routes through DAG and produces orbital_elu_ expressions
  expect_true(any(grepl("orbital_elu_", names(orb_obj))))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})


test_that("keras3 Functional model with ELU layer (alpha = 0.5) expression uses custom alpha", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L)(inp)
  x <- k$layers$ELU(alpha = 0.5)(x)
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
  # ELU with alpha=0.5: orbital_elu_ expressions must encode the custom alpha
  # (0.5 is exactly representable so format_numeric gives "0.5")
  expect_true(any(grepl("orbital_elu_", names(orb_obj))))
  hidden_exprs <- orb_obj[grepl("orbital_elu_", names(orb_obj))]
  expect_true(any(grepl("0.5", hidden_exprs, fixed = TRUE)))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

# ── Softmax max-stabilisation: overflow safety (R audit fix #3) ──────────────
