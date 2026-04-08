# Integration tests for keras3 structural layer handlers
# These tests are skipped when required packages are not installed.


test_that("keras3 Functional model with Concatenate layer translates correctly", {
  skip_if_no_keras3()

  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  branch_a <- k$layers$Dense(4L, activation = "relu")(inp)
  branch_b <- k$layers$Dense(4L, activation = "relu")(inp)
  x <- k$layers$Concatenate()(list(branch_a, branch_b))
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inp, out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 3L, verbose = 0L)

  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = paste0("x", 1:4)
  )
  expect_true(is.character(orb_obj))
  expect_named(orb_obj, ".pred", ignore.order = TRUE)
})

# ── new features: standalone Activation, AveragePooling1D, MaxPooling1D,
#                 GlobalSumPooling1D, multi-output (R#34, R#35, R#36, R#32) ───


test_that("keras3 Permute (transpose) layer predictions match keras3 predict", {
  .keras_skip()
  k <- reticulate::import("keras")
  T_len <- 3L
  C_in <- 2L
  inp <- k$Input(shape = list(T_len, C_in))
  # Permute (2, 1) swaps time and channel axes -> output shape (C_in, T_len)
  x <- k$layers$Permute(dims = list(2L, 1L))(inp)
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


test_that("keras3 RepeatVector predictions match keras3 predict", {
  .keras_skip()
  k <- reticulate::import("keras")
  n_feat <- 3L
  n_rep <- 4L
  inp <- k$Input(shape = list(n_feat))
  x <- k$layers$Dense(n_feat)(inp)
  x <- k$layers$RepeatVector(n_rep)(x) # (batch, n_rep, n_feat)
  x <- k$layers$Flatten()(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(30), nrow = 10, ncol = n_feat)
  model$fit(x_mat, rnorm(10), epochs = 2L, verbose = 0L)

  df <- as.data.frame(x_mat)
  names(df) <- paste0("x", 1:n_feat)
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = paste0("x", 1:n_feat)
  )
  expect_equal(
    predict(orb_obj, df)$.pred,
    as.numeric(model$predict(x_mat, verbose = 0L)),
    tolerance = 1e-4
  )
})

# ── Embedding ─────────────────────────────────────────────────────────────────


test_that("keras3 Embedding layer predictions match keras3 predict", {
  .keras_skip()
  k <- reticulate::import("keras")
  vocab_size <- 10L
  embed_dim <- 4L
  T_len <- 3L

  inp <- k$Input(shape = list(T_len), dtype = "int32")
  x <- k$layers$Embedding(vocab_size, embed_dim)(inp)
  x <- k$layers$Flatten()(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  n_row <- 20L
  x_int <- matrix(
    sample(0L:(vocab_size - 1L), n_row * T_len, replace = TRUE),
    nrow = n_row,
    ncol = T_len
  )
  model$fit(x_int, rnorm(n_row), epochs = 3L, verbose = 0L)

  feature_names <- paste0("tok", seq_len(T_len))
  df <- as.data.frame(x_int)
  names(df) <- feature_names
  orb_obj <- orbital(model, mode = "regression", feature_names = feature_names)
  expect_equal(
    predict(orb_obj, df)$.pred,
    as.numeric(model$predict(x_int, verbose = 0L)),
    tolerance = 1e-4
  )
})

# ── TimeDistributed ───────────────────────────────────────────────────────────


test_that("keras3 TimeDistributed(Dense) predictions match keras3 predict", {
  .keras_skip()
  k <- reticulate::import("keras")
  T_len <- 4L
  C_in <- 3L
  units <- 5L

  inp <- k$Input(shape = list(T_len, C_in))
  x <- k$layers$TimeDistributed(k$layers$Dense(units, activation = "relu"))(inp)
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

