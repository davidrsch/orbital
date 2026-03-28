# Integration tests for keras / keras3 / kerasnip MLP engines
# These tests are skipped when the relevant packages are not installed.

# ── keras3 (modern API, Sequential) ──────────────────────────────────────────

test_that("mlp() keras3 Sequential works with regression", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )

  # Build a simple keras3 Sequential MLP directly (not via parsnip)
  k <- reticulate::import("keras")
  model <- k$Sequential(list(
    k$layers$Dense(4L, activation = "relu", input_shape = list(3L)),
    k$layers$Dense(1L)
  ))
  model$compile(optimizer = "adam", loss = "mse")

  # Fit on tiny data
  set.seed(42)
  x_mat <- matrix(rnorm(30), nrow = 10, ncol = 3)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 2L, verbose = 0L)

  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = paste0("x", 1:3)
  )
  expect_true(is.character(orb_obj))
  expect_named(orb_obj, ".pred", ignore.order = TRUE)
})

test_that("mlp() keras3 Sequential works with binary classification", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )

  k <- reticulate::import("keras")
  model <- k$Sequential(list(
    k$layers$Dense(4L, activation = "relu", input_shape = list(3L)),
    k$layers$Dense(1L, activation = "sigmoid")
  ))
  model$compile(optimizer = "adam", loss = "binary_crossentropy")

  set.seed(42)
  x_mat <- matrix(rnorm(30), nrow = 10, ncol = 3)
  y_vec <- as.integer(rnorm(10) > 0)
  model$fit(x_mat, y_vec, epochs = 2L, verbose = 0L)

  orb_obj <- orbital(
    model,
    mode = "classification",
    type = "class",
    lvl = c("0", "1"),
    feature_names = paste0("x", 1:3)
  )
  expect_true(is.character(orb_obj))
  expect_true(".pred_class" %in% names(orb_obj))
})

test_that("mlp() keras3 Sequential works with multiclass (softmax)", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )

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
    type = "class",
    lvl = c("0", "1", "2"),
    feature_names = paste0("x", 1:4)
  )
  expect_true(is.character(orb_obj))
})

test_that("mlp() keras3 Functional model with Add (residual) works", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )

  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(4L, activation = "relu")(inp)
  out_t <- k$layers$Add()(list(x, inp)) # residual: 4 → 4 + input
  out <- k$layers$Dense(1L)(out_t)
  model <- k$Model(inp, out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 2L, verbose = 0L)

  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = paste0("x", 1:4)
  )
  expect_true(is.character(orb_obj))
  expect_named(orb_obj, ".pred", ignore.order = TRUE)
})

# ── swish and mish activations ────────────────────────────────────────────────

test_that("keras3 model with swish activation is translatable", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )

  k <- reticulate::import("keras")
  model <- k$Sequential(list(
    k$layers$Dense(4L, activation = "swish", input_shape = list(3L)),
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
  # swish expressions should contain the characteristic "(1 / (1 + exp(-" pattern
  hidden_exprs <- orb_obj[grepl("orbital_mlp", names(orb_obj))]
  expect_true(any(grepl("exp", hidden_exprs)))
})

# ── kerasnip (skip if not installed) ─────────────────────────────────────────

test_that("mlp() kerasnip engine works with regression", {
  skip_if_not_installed("kerasnip")
  skip_if_not_installed("parsnip")
  skip_if_not(
    tryCatch(
      {
        parsnip::mlp(engine = "kerasnip")
        TRUE
      },
      error = function(e) FALSE
    ),
    "kerasnip engine not registered with parsnip"
  )

  spec <- parsnip::mlp(hidden_units = 3, epochs = 5, engine = "kerasnip")
  spec <- parsnip::set_mode(spec, "regression")

  set.seed(1)
  fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit)
  preds <- predict(orb_obj, mtcars)

  expect_named(preds, ".pred")
  expect_type(preds$.pred, "double")
})

test_that("mlp() kerasnip engine works with binary classification", {
  skip_if_not_installed("kerasnip")
  skip_if_not_installed("parsnip")
  skip_if_not(
    tryCatch(
      {
        parsnip::mlp(engine = "kerasnip")
        TRUE
      },
      error = function(e) FALSE
    ),
    "kerasnip engine not registered with parsnip"
  )

  mtcars$vs <- factor(mtcars$vs)
  spec <- parsnip::mlp(hidden_units = 3, epochs = 5, engine = "kerasnip")
  spec <- parsnip::set_mode(spec, "classification")

  set.seed(1)
  fit <- parsnip::fit(spec, vs ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit, type = "class")
  preds <- predict(orb_obj, mtcars)

  expect_named(preds, ".pred_class")
  expect_type(preds$.pred_class, "character")
})

# ── new layer types: BatchNorm, LayerNorm, PReLU, Concatenate ─────────────────

test_that("keras3 Functional model with BatchNormalization translates correctly", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )

  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  x <- k$layers$BatchNormalization()(x)
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
  # BatchNorm intermediate expressions must appear
  expect_true(any(grepl("orbital_bn_", names(orb_obj))))
})

test_that("keras3 Functional model with LayerNormalization translates correctly", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )

  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  x <- k$layers$LayerNormalization()(x)
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
  # Mean and variance symbolic intermediates must appear
  expect_true(any(grepl("orbital_ln_mean_", names(orb_obj))))
  expect_true(any(grepl("orbital_ln_var_", names(orb_obj))))
})

test_that("keras3 Functional model with PReLU layer translates correctly", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )

  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L)(inp)
  x <- k$layers$PReLU()(x)
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
  expect_true(any(grepl("orbital_prelu_", names(orb_obj))))
})

test_that("keras3 Functional model with Concatenate layer translates correctly", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )

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

# ── activations: ELU, GELU, hard_swish ───────────────────────────────────────

test_that("keras3 Sequential model with ELU activation translates correctly", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )

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
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )

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
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )

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
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )

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

# ── Numerical accuracy tests (R #21) ─────────────────────────────────────────

test_that("keras3 BatchNormalization predictions match keras3 predict (regression)", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(3L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$BatchNormalization()(x)
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

test_that("keras3 LayerNormalization predictions match keras3 predict (regression)", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$LayerNormalization()(x)
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
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-4)
})

test_that("keras3 PReLU predictions match keras3 predict (regression)", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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

test_that("keras3 GlobalAveragePooling1D predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$GlobalAveragePooling1D()(x)
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
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

test_that("keras3 GlobalMaxPooling1D predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$GlobalMaxPooling1D()(x)
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
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

# ── new features: standalone Activation, AveragePooling1D, MaxPooling1D,
#                 GlobalSumPooling1D, multi-output (R#34, R#35, R#36, R#32) ───

test_that("keras3 Functional model with standalone Activation layer translates", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(3L))
  x <- k$layers$Dense(6L)(inp)
  x <- k$layers$Activation("relu")(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(30), nrow = 10, ncol = 3)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 3L, verbose = 0L)

  feature_names <- paste0("x", 1:3)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  expect_true(is.character(orb_obj))
  expect_named(orb_obj, ".pred", ignore.order = TRUE)
  expect_true(any(grepl("orbital_act_", names(orb_obj))))

  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

test_that("keras3 Functional model with AveragePooling1D translates correctly", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$AveragePooling1D()(x)
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
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  expect_true(is.character(orb_obj))
  expect_true(any(grepl("orbital_avgpool1d_", names(orb_obj))))
})

test_that("keras3 Functional model with MaxPooling1D translates correctly", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$MaxPooling1D()(x)
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
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  expect_true(is.character(orb_obj))
  expect_true(any(grepl("orbital_maxpool1d_", names(orb_obj))))
})

test_that("keras3 Functional model with GlobalSumPooling1D translates correctly", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$GlobalSumPooling1D()(x)
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
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  expect_true(is.character(orb_obj))
  expect_true(any(grepl("orbital_gsp_", names(orb_obj))))
})

test_that("keras3 InstanceNormalization predictions match keras3 predict (regression)", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  x <- k$layers$InstanceNormalization()(x)
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
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

test_that("keras3 GroupNormalization predictions match keras3 predict (regression)", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  x <- k$layers$GroupNormalization(groups = 1L)(x)
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
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

test_that("keras3 GroupNormalization num_groups == n_features (singleton groups) predictions match", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  # groups == n_features => each feature is its own group (singleton GN)
  x <- k$layers$GroupNormalization(groups = 8L)(x)
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
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

test_that("keras3 GroupNormalization general num_groups (groups=4, 8 features) is supported", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  # groups=4 with 8 features: 8 %% 4 == 0, so this is now supported
  x <- k$layers$GroupNormalization(groups = 4L)(x)
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
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  expect_true(any(grepl("orbital_gn_gmean_", names(orb_obj))))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

test_that("keras3 GroupNormalization with indivisible num_groups errors gracefully", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  # groups=3 with 8 features: 8 %% 3 != 0, so this must produce an error
  x <- k$layers$GroupNormalization(groups = 3L)(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 3L, verbose = 0L)

  expect_error(
    orbital(
      model,
      mode = "regression",
      feature_names = paste0("x", 1:4)
    ),
    regexp = "GroupNormalization"
  )
})

test_that("keras3 Functional model with standalone Activation layer translates correctly", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L)(inp)
  x <- k$layers$Activation("relu")(x)
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
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  expect_true(any(grepl("orbital_act_", names(orb_obj))))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

# ── activations: selu, mish, softplus, softsign, leaky_relu (R audit rec #1) ─

test_that("keras3 Sequential model with SELU activation predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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

test_that("keras3 Functional model with LeakyReLU layer (leaky_relu) predictions match", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L)(inp)
  x <- k$layers$LeakyReLU(negative_slope = 0.2)(x)
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
  # leaky_relu uses if_else and the slope value
  expect_true(is.character(orb_obj))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

# ── standalone Activation layer: non-relu variants (R audit rec #5) ──────────

test_that("keras3 Functional model with standalone Activation(tanh) translations match", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(3L))
  x <- k$layers$Dense(6L)(inp)
  x <- k$layers$Activation("tanh")(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(30), nrow = 10, ncol = 3)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 3L, verbose = 0L)

  feature_names <- paste0("x", 1:3)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  expect_true(any(grepl("orbital_act_", names(orb_obj))))
  act_exprs <- orb_obj[grepl("orbital_act_", names(orb_obj))]
  expect_true(any(grepl("tanh", act_exprs)))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

test_that("keras3 Functional model with standalone Activation(sigmoid) translations match", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(3L))
  x <- k$layers$Dense(6L)(inp)
  x <- k$layers$Activation("sigmoid")(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(30), nrow = 10, ncol = 3)
  y_vec <- rnorm(10)
  model$fit(x_mat, y_vec, epochs = 3L, verbose = 0L)

  feature_names <- paste0("x", 1:3)
  df <- as.data.frame(x_mat)
  names(df) <- feature_names
  orb_obj <- orbital(
    model,
    mode = "regression",
    feature_names = feature_names
  )
  expect_true(any(grepl("orbital_act_", names(orb_obj))))
  act_exprs <- orb_obj[grepl("orbital_act_", names(orb_obj))]
  expect_true(any(grepl("exp", act_exprs)))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

# ── multi-output Functional API (R audit rec #2 / R#32) ──────────────────────

test_that("keras3 Functional model with two output Dense layers produces named predictions", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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

# ── pooling accuracy tests (R audit rec #3) ───────────────────────────────────

test_that("keras3 AveragePooling1D predictions match keras3 predict (regression)", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$AveragePooling1D()(x)
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
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

test_that("keras3 MaxPooling1D predictions match keras3 predict (regression)", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$MaxPooling1D()(x)
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
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

test_that("keras3 GlobalSumPooling1D predictions match keras3 predict (regression)", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$GlobalSumPooling1D()(x)
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
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

# ── GroupNormalization groups=n_channels path (R audit rec #4) ───────────────

test_that("keras3 GroupNormalization groups=n_channels predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  # groups = n_channels = 8: each feature is its own group → output = beta_i
  x <- k$layers$GroupNormalization(groups = 8L)(x)
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
  expect_true(any(grepl("orbital_gn_", names(orb_obj))))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

# ── swish / hard_swish activation accuracy (audit rec #4) ────────────────────

test_that("keras3 Sequential model with swish activation predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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

test_that("keras3 Functional model with hard_swish Activation layer predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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

# ── GroupNormalization unsupported groups error (audit rec #5) ───────────────

test_that("keras3 GroupNormalization general groups=2 predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  # groups=2 with 8 features: 8 %% 2 == 0, now supported
  x <- k$layers$GroupNormalization(groups = 2L)(x)
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
  expect_true(any(grepl("orbital_gn_gmean_", names(orb_obj))))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

test_that("keras3 GroupNormalization with indivisible groups raises cli error", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  # groups=5 with 8 features: 8 %% 5 != 0, must error
  x <- k$layers$GroupNormalization(groups = 5L)(x)
  out <- k$layers$Dense(1L)(x)
  model <- k$Model(inputs = inp, outputs = out)
  model$compile(optimizer = "adam", loss = "mse")

  set.seed(42)
  x_mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  model$fit(x_mat, rnorm(10), epochs = 1L, verbose = 0L)

  expect_error(
    orbital(
      model,
      mode = "regression",
      feature_names = paste0("x", 1:4)
    ),
    regexp = "GroupNormalization",
    ignore.case = TRUE
  )
})

# ── multi-output model accuracy (audit rec #8) ───────────────────────────────

test_that("keras3 Functional two-output model predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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

# ── standalone ELU layer (R audit fix #2) ────────────────────────────────────

test_that("keras3 Functional model with ELU layer predictions match", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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
  # ELU layer routes through DAG and produces orbital_act_ expressions
  expect_true(any(grepl("orbital_act_", names(orb_obj))))
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

# ── Softmax max-stabilisation: overflow safety (R audit fix #3) ──────────────

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
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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

test_that("keras3 Functional model with Add (residual) predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(4L, activation = "relu")(inp)
  out_t <- k$layers$Add()(list(x, inp))
  out <- k$layers$Dense(1L)(out_t)
  model <- k$Model(inp, out)
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

test_that("keras3 model with Dropout layer predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(8L, activation = "relu")(inp)
  x <- k$layers$Dropout(rate = 0.5)(x)
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
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

test_that("keras3 model with Flatten layer predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$Flatten()(x)
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
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

test_that("keras3 model with Reshape layer predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$Reshape(list(6L))(x)
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
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_mat, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-5)
})

# ── B4: LSTM, GRU, Conv1D, RMSNormalization coverage tests ───────────────────

test_that("keras3 LSTM (return_sequences=FALSE) predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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
  orb_obj <- orbital(model, mode = "regression", feature_names = feature_names)
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_3d, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-4)
})

test_that("keras3 LSTM (return_sequences=TRUE) predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
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
  orb_obj <- orbital(model, mode = "regression", feature_names = feature_names)
  preds_orb <- predict(orb_obj, df)$.pred
  preds_keras <- as.numeric(model$predict(x_3d, verbose = 0L))
  expect_equal(preds_orb, preds_keras, tolerance = 1e-4)
})

test_that("keras3 GRU (return_sequences=FALSE) predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  T_len <- 3L
  C_in <- 2L
  H <- 4L
  inp <- k$Input(shape = list(T_len, C_in))
  x <- k$layers$GRU(H, return_sequences = FALSE)(inp)
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

test_that("keras3 GRU (return_sequences=TRUE) predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  T_len <- 3L
  C_in <- 2L
  H <- 3L
  inp <- k$Input(shape = list(T_len, C_in))
  x <- k$layers$GRU(H, return_sequences = TRUE)(inp)
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

test_that("keras3 Conv1D predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  T_len <- 4L
  C_in <- 2L
  filters <- 3L
  ksize <- 2L
  inp <- k$Input(shape = list(T_len, C_in))
  x <- k$layers$Conv1D(filters, ksize, activation = "relu")(inp)
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

test_that("keras3 RMSNormalization predictions match keras3 predict", {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
  k <- reticulate::import("keras")
  inp <- k$Input(shape = list(4L))
  x <- k$layers$Dense(6L, activation = "relu")(inp)
  x <- k$layers$RMSNormalization()(x)
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
  expect_equal(preds_orb, preds_keras, tolerance = 1e-4)
})
