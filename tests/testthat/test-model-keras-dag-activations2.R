# Integration tests for keras3 Functional models: celu, hardtanh, log_sigmoid + softmax stability.
# These tests are skipped when required packages are not installed.

test_that("Softmax orbital expressions are numerically stable for large logits", {
  # Direct expression-level test: verifies the max-stabilised pattern used by
  # orbital_keras_dag_impl() for standalone Softmax layers does not produce
  # NaN when logits are large enough to overflow exp() without stabilisation.
  df <- data.frame(a = 700, b = 0, c = 0)

  result <- dplyr::mutate(
    df,
    sm_max = do.call(pmax, list(a, b, c)),
    sm_sum = (exp(a - sm_max) + exp(b - sm_max) + exp(c - sm_max)),
    p1 = exp(a - sm_max) / sm_sum,
    p2 = exp(b - sm_max) / sm_sum,
    p3 = exp(c - sm_max) / sm_sum
  )

  expect_false(is.nan(result$p1))
  expect_false(is.nan(result$p2))
  expect_false(is.nan(result$p3))
  expect_equal(result$p1 + result$p2 + result$p3, 1.0, tolerance = 1e-10)
  # With logits (700, 0, 0) the first class should have probability ≈ 1
  expect_equal(result$p1, 1.0, tolerance = 1e-5)
})

# ── activations: celu, hardtanh, log_sigmoid via standalone Activation layer ─

test_that("keras3 Functional model with Activation('celu') predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L)(inp)
  x <- k$layers$Activation("celu")(x)
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
  # CELU expression uses dplyr::if_else and exp (not pmax/pmin)
  hidden_exprs <- orb_obj[grepl("orbital_act_", names(orb_obj))]
  expect_true(any(grepl("if_else", hidden_exprs)))
  expect_true(any(grepl("exp", hidden_exprs)))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-4)
})


test_that("keras3 Functional model with Activation('hardtanh') predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L)(inp)
  x <- k$layers$Activation("hardtanh")(x)
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
  # HardTanh expression clips to [-1, 1]
  hidden_exprs <- orb_obj[grepl("orbital_act_", names(orb_obj))]
  expect_true(any(grepl("dplyr::if_else", hidden_exprs)))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})


test_that("keras3 Functional model with Activation('log_sigmoid') predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L)(inp)
  x <- k$layers$Activation("log_sigmoid")(x)
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
  # LogSigmoid = -log(1 + exp(-x))
  hidden_exprs <- orb_obj[grepl("orbital_act_", names(orb_obj))]
  expect_true(any(grepl("-log\\(1", hidden_exprs)))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

# ── Add (residual) / Dropout / Flatten / Reshape accuracy (audit rec #9) ────
