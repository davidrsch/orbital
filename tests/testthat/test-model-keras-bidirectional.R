# Integration tests for keras3 Bidirectional RNN layers
# These tests are skipped when required packages are not installed.

test_that("keras3 Bidirectional(LSTM, concat) predictions match keras3 predict", {
    skip_if_no_keras3()
    k <- reticulate::import("keras")
    T_len <- 3L
    C_in <- 2L
    H <- 3L
    inp <- k$Input(shape = list(T_len, C_in))
    x <- k$layers$Bidirectional(
        k$layers$LSTM(H, return_sequences = FALSE),
        merge_mode = "concat"
    )(inp)
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

test_that("keras3 Bidirectional(GRU, concat) predictions match keras3 predict", {
    skip_if_no_keras3()
    k <- reticulate::import("keras")
    T_len <- 3L
    C_in <- 2L
    H <- 3L
    inp <- k$Input(shape = list(T_len, C_in))
    x <- k$layers$Bidirectional(
        k$layers$GRU(H, return_sequences = FALSE),
        merge_mode = "concat"
    )(inp)
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

test_that("keras3 Bidirectional(LSTM, sum) predictions match keras3 predict", {
    skip_if_no_keras3()
    k <- reticulate::import("keras")
    T_len <- 3L
    C_in <- 2L
    H <- 3L
    inp <- k$Input(shape = list(T_len, C_in))
    x <- k$layers$Bidirectional(
        k$layers$LSTM(H, return_sequences = FALSE),
        merge_mode = "sum"
    )(inp)
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

test_that("keras3 Bidirectional(LSTM, ave) predictions match keras3 predict", {
    skip_if_no_keras3()
    k <- reticulate::import("keras")
    T_len <- 3L
    C_in <- 2L
    H <- 3L
    inp <- k$Input(shape = list(T_len, C_in))
    x <- k$layers$Bidirectional(
        k$layers$LSTM(H, return_sequences = FALSE),
        merge_mode = "ave"
    )(inp)
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

test_that("keras3 Bidirectional(LSTM, return_sequences=TRUE) predictions match keras3 predict", {
    skip_if_no_keras3()
    k <- reticulate::import("keras")
    T_len <- 3L
    C_in <- 2L
    H <- 2L
    inp <- k$Input(shape = list(T_len, C_in))
    x <- k$layers$Bidirectional(
        k$layers$LSTM(H, return_sequences = TRUE),
        merge_mode = "concat"
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

test_that("keras3 Bidirectional(GRU, return_sequences=TRUE) predictions match keras3 predict", {
    skip_if_no_keras3()
    k <- reticulate::import("keras")
    T_len <- 3L
    C_in <- 2L
    H <- 2L
    inp <- k$Input(shape = list(T_len, C_in))
    x <- k$layers$Bidirectional(
        k$layers$GRU(H, return_sequences = TRUE),
        merge_mode = "concat"
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

test_that("keras3 Bidirectional(LSTM, mul) predictions match keras3 predict", {
    .keras_skip()
    k <- reticulate::import("keras")
    T_len <- 3L
    C_in <- 2L
    H <- 3L
    inp <- k$Input(shape = list(T_len, C_in))
    x <- k$layers$Bidirectional(
        k$layers$LSTM(H, return_sequences = FALSE),
        merge_mode = "mul"
    )(inp)
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
    orb_obj <- orbital(
        model,
        mode = "regression",
        feature_names = feature_names
    )
    expect_equal(
        predict(orb_obj, df)$.pred,
        as.numeric(model$predict(x_3d, verbose = 0L)),
        tolerance = 1e-4
    )
})

test_that("keras3 Bidirectional(LSTM, merge_mode=None) falls back to concat", {
    skip_if_no_keras3()
    k <- reticulate::import("keras")
    T_len <- 3L
    C_in <- 2L
    units <- 4L
    inp <- k$Input(shape = list(T_len, C_in))
    x <- k$layers$Bidirectional(
        k$layers$LSTM(units, return_sequences = FALSE),
        merge_mode = reticulate::r_to_py(NULL)
    )(inp)
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
    # merge_mode=None falls back to concat silently
    orb_obj <- orbital(
        model,
        mode = "regression",
        feature_names = feature_names
    )
    expect_s3_class(orb_obj, "orbital_class")
})

# ── Wave 1A regression: Bidirectional(LSTM) 2-row bias colSums fix ────────────

test_that("Bidirectional(LSTM, concat) with explicit 2-row bias matrices gives correct predictions (Wave 1A regression)", {
    skip_if_no_keras3()
    k <- reticulate::import("keras")
    T_len <- 3L
    C_in <- 2L
    H <- 4L
    n_row <- 5L

    inp <- k$Input(shape = list(T_len, C_in))
    x <- k$layers$Bidirectional(
        k$layers$LSTM(H, return_sequences = FALSE),
        merge_mode = "concat"
    )(inp)
    out <- k$layers$Dense(1L)(x)
    model <- k$Model(inputs = inp, outputs = out)
    model$compile(optimizer = "adam", loss = "mse")

    # Fit briefly to allocate weight tensors with keras-assigned shapes
    set.seed(42)
    x_flat <- matrix(
        rnorm(n_row * T_len * C_in),
        nrow = n_row,
        ncol = T_len * C_in
    )
    x_3d <- array(x_flat, dim = c(n_row, T_len, C_in))
    model$fit(x_3d, rnorm(n_row), epochs = 1L, verbose = 0L)

    # Verify that keras3 Bidirectional LSTM stores bias as 2-row matrix
    curr_wts <- model$get_weights()
    skip_if(
        !is.matrix(curr_wts[[3L]]) || nrow(curr_wts[[3L]]) != 2L,
        "Keras version does not produce 2-row LSTM bias; skipping Wave 1A 2-row bias regression test"
    )

    # Replace bias entries with controlled (2, 4H) values to exercise colSums fix
    set.seed(99)
    curr_wts[[3L]] <- matrix(rnorm(2L * 4L * H), nrow = 2L, ncol = 4L * H)
    curr_wts[[6L]] <- matrix(rnorm(2L * 4L * H), nrow = 2L, ncol = 4L * H)
    model$set_weights(curr_wts)

    feature_names <- paste0("x", seq_len(T_len * C_in))
    df <- as.data.frame(x_flat)
    names(df) <- feature_names

    orb_obj <- orbital(
        model,
        mode = "regression",
        feature_names = feature_names
    )
    expect_equal(
        predict(orb_obj, df)$.pred,
        as.numeric(model$predict(x_3d, verbose = 0L)),
        tolerance = 1e-4
    )
})

test_that("keras3 Bidirectional with non-LSTM/GRU inner layer raises cli_abort", {
    skip_if_no_keras3()
    k <- reticulate::import("keras")
    inp <- k$Input(shape = list(3L, 2L))
    x <- k$layers$Bidirectional(
        k$layers$SimpleRNN(4L)
    )(inp)
    out <- k$layers$Dense(1L)(x)
    model <- k$Model(inputs = inp, outputs = out)
    model$compile(optimizer = "adam", loss = "mse")
    expect_error(
        orbital(model, mode = "regression", feature_names = paste0("x", 1:6)),
        regexp = "not LSTM or GRU"
    )
})
