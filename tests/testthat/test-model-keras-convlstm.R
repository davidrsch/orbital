# Integration tests for keras3 ConvLSTM1D layers
# These tests are skipped when required packages are not installed.

test_that("keras3 ConvLSTM1D (return_sequences=FALSE) predictions match keras3 predict", {
    skip_if_no_keras3()
    k <- reticulate::import("keras")
    T_len <- 3L
    S_len <- 1L
    C_in <- 2L
    F_filt <- 4L
    inp <- k$Input(shape = list(T_len, S_len, C_in))
    x <- k$layers$ConvLSTM1D(
        filters = F_filt,
        kernel_size = 3L,
        padding = "same",
        return_sequences = FALSE
    )(inp)
    x <- k$layers$Flatten()(x)
    out <- k$layers$Dense(1L)(x)
    model <- k$Model(inputs = inp, outputs = out)
    model$compile(optimizer = "adam", loss = "mse")

    set.seed(42)
    n_row <- 10L
    x_flat <- matrix(
        rnorm(n_row * T_len * S_len * C_in),
        nrow = n_row,
        ncol = T_len * S_len * C_in
    )
    y_vec <- rnorm(n_row)
    x_4d <- array(x_flat, dim = c(n_row, T_len, S_len, C_in))
    model$fit(x_4d, y_vec, epochs = 3L, verbose = 0L)

    feature_names <- paste0("x", seq_len(T_len * S_len * C_in))
    df <- as.data.frame(x_flat)
    names(df) <- feature_names
    orb_obj <- orbital(
        model,
        mode = "regression",
        feature_names = feature_names
    )
    preds_orb <- predict(orb_obj, df)$.pred
    preds_keras <- as.numeric(model$predict(x_4d, verbose = 0L))
    expect_equal(preds_orb, preds_keras, tolerance = 1e-4)
})

test_that("keras3 ConvLSTM1D (return_sequences=TRUE) predictions match keras3 predict", {
    skip_if_no_keras3()
    k <- reticulate::import("keras")
    T_len <- 3L
    S_len <- 1L
    C_in <- 2L
    F_filt <- 3L
    inp <- k$Input(shape = list(T_len, S_len, C_in))
    x <- k$layers$ConvLSTM1D(
        filters = F_filt,
        kernel_size = 3L,
        padding = "same",
        return_sequences = TRUE
    )(inp)
    x <- k$layers$Flatten()(x)
    out <- k$layers$Dense(1L)(x)
    model <- k$Model(inputs = inp, outputs = out)
    model$compile(optimizer = "adam", loss = "mse")

    set.seed(42)
    n_row <- 10L
    x_flat <- matrix(
        rnorm(n_row * T_len * S_len * C_in),
        nrow = n_row,
        ncol = T_len * S_len * C_in
    )
    y_vec <- rnorm(n_row)
    x_4d <- array(x_flat, dim = c(n_row, T_len, S_len, C_in))
    model$fit(x_4d, y_vec, epochs = 3L, verbose = 0L)

    feature_names <- paste0("x", seq_len(T_len * S_len * C_in))
    df <- as.data.frame(x_flat)
    names(df) <- feature_names
    orb_obj <- orbital(
        model,
        mode = "regression",
        feature_names = feature_names
    )
    preds_orb <- predict(orb_obj, df)$.pred
    preds_keras <- as.numeric(model$predict(x_4d, verbose = 0L))
    expect_equal(preds_orb, preds_keras, tolerance = 1e-4)
})

test_that("keras3 ConvLSTM1D stateful=TRUE raises cli_abort", {
    skip_if_no_keras3()
    k <- reticulate::import("keras")
    T_len <- 3L
    S_len <- 1L
    C_in <- 2L
    F_filt <- 4L
    inp <- k$Input(shape = list(T_len, S_len, C_in))
    x <- k$layers$ConvLSTM1D(
        filters = F_filt,
        kernel_size = 3L,
        padding = "same",
        return_sequences = FALSE,
        stateful = TRUE
    )(inp)
    x <- k$layers$Flatten()(x)
    out <- k$layers$Dense(1L)(x)
    model <- k$Model(inputs = inp, outputs = out)

    feature_names <- paste0("x", seq_len(T_len * S_len * C_in))
    expect_error(
        orbital(model, mode = "regression", feature_names = feature_names),
        "stateful"
    )
})
