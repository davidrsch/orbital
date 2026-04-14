# Integration tests for keras3 EinsumDense in DAG (Functional) models.
# These tests are skipped when required packages are not installed.

test_that("keras3 EinsumDense intermediate layer (ab,bc->ac, relu) predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$EinsumDense(
    equation = "ab,bc->ac",
    output_shape = 6L,
    activation = "relu"
  )(inp)
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
  expect_true(any(grepl("orbital_einsumdense_", names(orb_obj))))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-4)
})


test_that("keras3 EinsumDense as output layer (ab,bc->ac, BUG-1 fix) predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  out <- k$layers$EinsumDense(
    equation = "ab,bc->ac",
    output_shape = 1L,
    activation = "linear"
  )(x)
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


test_that("keras3 EinsumDense time-distributed (abc,cd->abd) predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  T_len <- 3L
  C_in <- 5L
  D_out <- 4L

  inp <- k$Input(shape = list(T_len, C_in))
  x <- k$layers$EinsumDense(
    equation = "abc,cd->abd",
    output_shape = list(T_len, D_out)
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
  y_vec <- rnorm(n_row)
  model$fit(x_3d, y_vec, epochs = 3L, verbose = 0L)

  feature_names <- paste0("x", seq_len(T_len * C_in))
  df <- as.data.frame(x_flat)
  names(df) <- feature_names
  orb_obj <- orbital(model, mode = "regression", feature_names = feature_names)
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_3d, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-4)
})
