# Integration tests for keras3 EinsumDense layers
# These tests are skipped when required packages are not installed.

test_that("keras3 EinsumDense (ab,bc->ac) predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  x <- k$layers$EinsumDense("ab,bc->ac", output_shape = 6L, bias_axes = "c")(x)
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
  orb_obj <- orbital(model, mode = "regression", feature_names = feature_names)
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

# ── EinsumDense with non-default activation alpha ─────────────────────────────

test_that("keras3 EinsumDense with LeakyReLU(alpha=0.2) predictions match keras3 predict", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  x <- k$layers$EinsumDense(
    "ab,bc->ac",
    output_shape = 6L,
    bias_axes = "c",
    activation = k$layers$LeakyReLU(negative_slope = 0.2)
  )(x)
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
  orb_obj <- orbital(model, mode = "regression", feature_names = feature_names)
  expect_equal(
    predict(orb_obj, df)$.pred,
    as.numeric(model$predict(x_mat, verbose = 0L)),
    tolerance = 1e-5
  )
})

# ── Wave 1A regression: Sequential Dense(use_bias=FALSE) weight-counting fix ──

test_that("Sequential Dense(use_bias=FALSE) + Dense produces correct predictions (Wave 1A regression)", {
  skip_if_no_keras3()
  k <- reticulate::import("keras")

  model <- k$Sequential(list(
    k$Input(shape = list(3L)),
    k$layers$Dense(units = 8L, use_bias = FALSE, activation = "relu"),
    k$layers$Dense(units = 1L)
  ))

  # Wave 1A fix: per-layer weight counting instead of length(all_weights)/2.
  # Layer 0 (use_bias=FALSE) has 1 weight; layer 1 has 2 weights.
  set.seed(42)
  w1 <- matrix(rnorm(3L * 8L), nrow = 3L, ncol = 8L) # kernel (3 × 8)
  w2 <- matrix(rnorm(8L * 1L), nrow = 8L, ncol = 1L) # kernel (8 × 1)
  b2 <- array(rnorm(1L), dim = 1L) # bias  (1,)
  model$set_weights(list(w1, w2, b2))

  feature_names <- c("a", "b", "c")
  df <- data.frame(a = c(1, 2, 3), b = c(-1, 0, 1), c = c(0.5, -0.5, 2))

  orb_obj <- orbital(model, mode = "regression", feature_names = feature_names)
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(
    model$predict(as.matrix(df[, feature_names]), verbose = 0L)
  )
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})
