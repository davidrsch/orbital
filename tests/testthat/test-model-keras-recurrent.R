# Integration tests for keras3 SimpleRNN and masking-related layers.
# LSTM → test-model-keras-lstm.R
# GRU → test-model-keras-gru.R
# Bidirectional → test-model-keras-bidirectional.R
# ConvLSTM1D → test-model-keras-convlstm.R

# ── SimpleRNN ─────────────────────────────────────────────────────────────────

test_that("keras3 SimpleRNN (return_sequences=FALSE) predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  T_len <- 3L
  C_in <- 2L
  H <- 4L
  inp <- k$Input(shape = list(T_len, C_in))
  x <- k$layers$SimpleRNN(H, return_sequences = FALSE)(inp)
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

test_that("keras3 SimpleRNN (return_sequences=TRUE) predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  T_len <- 3L
  C_in <- 2L
  H <- 3L
  inp <- k$Input(shape = list(T_len, C_in))
  x <- k$layers$SimpleRNN(H, return_sequences = TRUE)(inp)
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

# ── .detect_masking_upstream unit tests (R-G: masking detection) ─────────────

test_that(".detect_masking_upstream returns TRUE when direct parent is a masking layer", {
  topo <- list(
    lstm_1 = "masking",
    masking = "embedding_1",
    embedding_1 = character(0L)
  )
  expect_true(orbital:::.detect_masking_upstream("lstm_1", topo))
})

test_that(".detect_masking_upstream returns TRUE when masking layer is two hops upstream", {
  topo <- list(
    lstm_1 = "dense_1",
    dense_1 = "masking",
    masking = "input_1",
    input_1 = character(0L)
  )
  expect_true(orbital:::.detect_masking_upstream("lstm_1", topo))
})

test_that(".detect_masking_upstream returns FALSE when no masking layer is upstream", {
  topo <- list(
    lstm_1 = "embedding_1",
    embedding_1 = "input_1",
    input_1 = character(0L)
  )
  expect_false(orbital:::.detect_masking_upstream("lstm_1", topo))
})

test_that(".detect_masking_upstream returns FALSE for an empty topo_map entry", {
  topo <- list(lstm_1 = character(0L))
  expect_false(orbital:::.detect_masking_upstream("lstm_1", topo))
})

test_that(".detect_masking_upstream returns FALSE for an unknown layer name", {
  topo <- list(some_layer = "input_1", input_1 = character(0L))
  expect_false(orbital:::.detect_masking_upstream("nonexistent_layer", topo))
})

test_that(".detect_masking_upstream is case-insensitive for masking layer prefix", {
  topo <- list(lstm_1 = "Masking_layer", Masking_layer = character(0L))
  expect_true(orbital:::.detect_masking_upstream("lstm_1", topo))
})
