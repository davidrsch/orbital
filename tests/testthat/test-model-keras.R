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
    expect_true(any(grepl("tanh", hidden_exprs)))
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
