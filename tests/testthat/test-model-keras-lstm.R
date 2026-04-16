# Integration tests for keras3 LSTM layers
# These tests are skipped when required packages are not installed.

test_that("keras3 LSTM (return_sequences=FALSE) predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  T_len <- 3L
  C_in <- 2L
  H <- 4L
  inp <- k$Input(shape = list(T_len, C_in))
  x <- k$layers$LSTM(H, return_sequences = FALSE)(inp)
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
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_3d, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-4)
})

test_that("keras3 LSTM (return_sequences=TRUE) predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  T_len <- 3L
  C_in <- 2L
  H <- 3L
  inp <- k$Input(shape = list(T_len, C_in))
  x <- k$layers$LSTM(H, return_sequences = TRUE)(inp)
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
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_3d, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-4)
})

test_that("keras3 stateful LSTM raises cli_abort", {
  .keras_skip()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(3L, 2L))
  x <- k$layers$LSTM(4L, stateful = TRUE)(inp)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  feature_names <- paste0("x", 1:6)
  expect_error(
    orbital(model, mode = "regression", feature_names = feature_names),
    "stateful"
  )
})

test_that("keras3 LSTM (go_backwards=TRUE) produces different orbital expressions than go_backwards=FALSE", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  T_len <- 3L
  C_in <- 2L
  H <- 2L

  make_lstm_model <- function(go_bwd) {
    inp <- k$Input(shape = list(T_len, C_in))
    x <- k$layers$LSTM(H, return_sequences = FALSE, go_backwards = go_bwd)(
      inp
    )
    out <- k$layers$Dense(1L)(x)
    mdl <- k$Model(inputs = inp, outputs = out)
    mdl$compile(optimizer = "adam", loss = "mse")
    mdl
  }

  set.seed(42)
  n_row <- 10L
  x_flat <- matrix(
    rnorm(n_row * T_len * C_in),
    nrow = n_row,
    ncol = T_len * C_in
  )
  x_3d <- array(x_flat, dim = c(n_row, T_len, C_in))

  model_fwd <- make_lstm_model(FALSE)
  model_bwd <- make_lstm_model(TRUE)
  # Fit forward model then copy identical weights to backward model so that any
  # expression difference is purely structural (loop order), not numerical.
  model_fwd$fit(x_3d, rnorm(n_row), epochs = 1L, verbose = 0L)
  model_bwd$set_weights(model_fwd$get_weights())

  feature_names <- paste0("x", seq_len(T_len * C_in))
  orb_fwd <- orbital(
    model_fwd,
    mode = "regression",
    feature_names = feature_names
  )
  orb_bwd <- orbital(
    model_bwd,
    mode = "regression",
    feature_names = feature_names
  )

  # Regression guard: go_backwards changes the time-step traversal order, so
  # the symbolic expression chain must differ between the two directions.
  expect_false(identical(unclass(orb_fwd), unclass(orb_bwd)))
})

test_that("keras3 LSTM (go_backwards=TRUE, return_sequences=FALSE) predictions match keras3", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  T_len <- 3L
  C_in <- 2L
  H <- 4L
  inp <- k$Input(shape = list(T_len, C_in))
  x <- k$layers$LSTM(H, return_sequences = FALSE, go_backwards = TRUE)(inp)
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
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_3d, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-4)
})

test_that("keras3 LSTM (go_backwards=TRUE, return_sequences=TRUE) predictions match keras3", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  T_len <- 3L
  C_in <- 2L
  H <- 3L
  inp <- k$Input(shape = list(T_len, C_in))
  x <- k$layers$LSTM(H, return_sequences = TRUE, go_backwards = TRUE)(inp)
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
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_3d, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-4)
})
