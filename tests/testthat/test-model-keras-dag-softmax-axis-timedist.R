# Test softmax axis validation and TimeDistributed edge cases for the keras3
# Functional DAG backend.  These tests gate H2 (softmax axis) and M6
# (TimeDistributed coverage) findings from the audit.

test_that("keras3 Softmax(axis = -1) Functional model matches keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$Dense(3L)(x)
  out <- k$layers$Softmax(axis = -1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "sparse_categorical_crossentropy")

  set.seed(1)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  y_vec <- as.integer(sample.int(3, 10, replace = TRUE)) - 1L
  model$fit(x_mat, y_vec, epochs = 3L, verbose = 0L)

  feature_names <- paste0("x", 1:4)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  orb_obj <- orbital(
    model,
    mode = "classification",
    type = "prob",
    lvl = c("a", "b", "c"),
    feature_names = feature_names
  )
  preds_orb <- predict(orb_obj, df)
  preds_keras <- model$predict(x_mat, verbose = 0L)
  expect_equal(
    as.numeric(as.matrix(preds_orb[, c(".pred_a", ".pred_b", ".pred_c")])),
    as.numeric(preds_keras),
    tolerance = 1e-5
  )
})


test_that("keras3 Softmax(axis = 1) is rejected with a clear error", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  # Sequence-shaped input so that axis = 1 is a non-last axis.
  inp <- k$Input(shape = list(4L, 3L))
  x <- k$layers$Dense(3L)(inp)
  out <- k$layers$Softmax(axis = 1L)(x)
  model <- k$Model(inputs = inp, outputs = out)

  feature_names <- paste0("x", 1:12)
  expect_error(
    orbital(
      model,
      mode = "classification",
      type = "prob",
      lvl = letters[1:3],
      feature_names = feature_names
    ),
    regexp = "axis"
  )
})


test_that("keras3 Activation('softmax') on Dense output matches keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$Dense(3L)(x)
  out <- k$layers$Activation("softmax")(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "sparse_categorical_crossentropy")

  set.seed(2)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  y_vec <- as.integer(sample.int(3, 10, replace = TRUE)) - 1L
  model$fit(x_mat, y_vec, epochs = 3L, verbose = 0L)

  feature_names <- paste0("x", 1:4)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  orb_obj <- orbital(
    model,
    mode = "classification",
    type = "prob",
    lvl = c("a", "b", "c"),
    feature_names = feature_names
  )
  preds_orb <- predict(orb_obj, df)
  preds_keras <- model$predict(x_mat, verbose = 0L)
  expect_equal(
    as.numeric(as.matrix(preds_orb[, c(".pred_a", ".pred_b", ".pred_c")])),
    as.numeric(preds_keras),
    tolerance = 1e-5
  )
})


# ── TimeDistributed coverage ─────────────────────────────────────────────────

test_that("keras3 TimeDistributed(Dense) with T = 1 matches keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  units <- 5L
  inp <- k$Input(shape = list(1L, 4L)) # T = 1 edge case
  x <- k$layers$TimeDistributed(k$layers$Dense(units, activation = "relu"))(
    inp
  )
  x <- k$layers$Flatten()(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(3)
  x_arr <- array(rnorm(10 * 1 * 4), dim = c(10L, 1L, 4L))
  y_vec <- rnorm(10)
  model$fit(x_arr, y_vec, epochs = 2L, verbose = 0L)

  feature_names <- paste0("x", 1:4)
  df <- as.data.frame(matrix(x_arr, nrow = 10, ncol = 4))
  names(df) <- feature_names
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  preds_orb <- predict(orb_obj, df)
  preds_keras <- model$predict(x_arr, verbose = 0L)
  expect_equal(preds_orb$.pred, as.numeric(preds_keras), tolerance = 1e-5)
})


test_that("keras3 TimeDistributed(Dense, return_sequences) matches keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  units <- 4L
  inp <- k$Input(shape = list(3L, 2L))
  x <- k$layers$TimeDistributed(k$layers$Dense(units))(inp)
  x <- k$layers$Flatten()(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(4)
  x_arr <- array(rnorm(8 * 3 * 2), dim = c(8L, 3L, 2L))
  y_vec <- rnorm(8)
  model$fit(x_arr, y_vec, epochs = 2L, verbose = 0L)

  feature_names <- paste0("x", 1:6)
  df <- as.data.frame(matrix(x_arr, nrow = 8, ncol = 6))
  names(df) <- feature_names
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  preds_orb <- predict(orb_obj, df)
  preds_keras <- model$predict(x_arr, verbose = 0L)
  expect_equal(preds_orb$.pred, as.numeric(preds_keras), tolerance = 1e-5)
})
