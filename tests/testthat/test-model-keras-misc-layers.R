# Integration tests for keras3 misc 1D spatial layers and EinsumDense ellipsis.
# Covers handlers implemented in orbital that were not previously tested with
# these specific parameter configurations:
#   .k3_zeropadding1d()          — asymmetric padding c(1, 2)
#   .k3_cropping1d()             — asymmetric cropping c(2, 1)
#   .k3_upsampling1d()           — size = 3
#   .k3_adaptiveaveragepooling1d() — direct 3-D input, output_size = 2
#   .k3_adaptivemaxpooling1d()     — direct 3-D input, output_size = 2
#   .k3_einsumdense() equation "...b,bc->...c"

# ── ZeroPadding1D (asymmetric padding c(1, 2)) ────────────────────────────────

test_that("keras3 ZeroPadding1D (asymmetric padding c(1,2)) predictions match keras3 predict", {
    .keras_skip()
    k <- reticulate::import("keras")
    T_len <- 3L
    C_in <- 2L
    inp <- k$Input(shape = list(T_len, C_in))
    x <- k$layers$ZeroPadding1D(padding = list(1L, 2L))(inp) # T_out = 3+1+2 = 6
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
    orb_obj <- orbital(
        model,
        mode = "regression",
        feature_names = feature_names
    )
    expect_true(any(grepl("orbital_zpad_", names(orb_obj))))
    expect_equal(
        predict(orb_obj, df)$.pred,
        as.numeric(model$predict(x_3d, verbose = 0L)),
        tolerance = 1e-4
    )
})

# ── Cropping1D (asymmetric cropping c(2, 1)) ──────────────────────────────────

test_that("keras3 Cropping1D (asymmetric cropping c(2,1)) predictions match keras3 predict", {
    .keras_skip()
    k <- reticulate::import("keras")
    T_len <- 6L
    C_in <- 2L
    inp <- k$Input(shape = list(T_len, C_in))
    x <- k$layers$Cropping1D(cropping = list(2L, 1L))(inp) # T_out = 6-2-1 = 3
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
    orb_obj <- orbital(
        model,
        mode = "regression",
        feature_names = feature_names
    )
    expect_true(any(grepl("orbital_crop_", names(orb_obj))))
    expect_equal(
        predict(orb_obj, df)$.pred,
        as.numeric(model$predict(x_3d, verbose = 0L)),
        tolerance = 1e-4
    )
})

# ── UpSampling1D (size = 3) ───────────────────────────────────────────────────

test_that("keras3 UpSampling1D (size = 3) predictions match keras3 predict", {
    .keras_skip()
    k <- reticulate::import("keras")
    T_len <- 3L
    C_in <- 2L
    inp <- k$Input(shape = list(T_len, C_in))
    x <- k$layers$UpSampling1D(size = 3L)(inp) # T_out = 3*3 = 9
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
    orb_obj <- orbital(
        model,
        mode = "regression",
        feature_names = feature_names
    )
    expect_true(any(grepl("orbital_up1d_", names(orb_obj))))
    expect_equal(
        predict(orb_obj, df)$.pred,
        as.numeric(model$predict(x_3d, verbose = 0L)),
        tolerance = 1e-4
    )
})

# ── AdaptiveAveragePooling1D (direct 3-D input, output_size = 2) ─────────────

test_that("keras3 AdaptiveAveragePooling1D (direct 3-D input, output_size=2) predictions match keras3 predict", {
    .keras_skip()
    k <- reticulate::import("keras")
    skip_if(
        is.null(tryCatch(
            k$layers$AdaptiveAveragePooling1D,
            error = function(e) NULL
        )),
        "AdaptiveAveragePooling1D not available in this Keras version"
    )
    T_len <- 6L
    C_in <- 4L
    inp <- k$Input(shape = list(T_len, C_in))
    x <- k$layers$AdaptiveAveragePooling1D(output_size = 2L)(inp)
    x <- k$layers$Flatten()(x)
    out <- k$layers$Dense(1L)(x)
    model <- k$Model(inputs = inp, outputs = out)
    model$compile(optimizer = "adam", loss = "mse")

    set.seed(42)
    n_row <- 10L
    x_arr <- array(rnorm(n_row * T_len * C_in), dim = c(n_row, T_len, C_in))
    model$fit(x_arr, rnorm(n_row), epochs = 2L, verbose = 0L)

    x_flat <- matrix(as.numeric(x_arr), nrow = n_row)
    feature_names <- paste0("x", seq_len(ncol(x_flat)))
    df <- as.data.frame(x_flat)
    names(df) <- feature_names
    orb_obj <- orbital(
        model,
        mode = "regression",
        feature_names = feature_names
    )
    expect_true(any(grepl("orbital_adaptiveavgpool1d_", names(orb_obj))))
    expect_equal(
        predict(orb_obj, df)$.pred,
        as.numeric(model$predict(x_arr, verbose = 0L)),
        tolerance = 1e-4
    )
})

# ── AdaptiveMaxPooling1D (direct 3-D input, output_size = 2) ─────────────────

test_that("keras3 AdaptiveMaxPooling1D (direct 3-D input, output_size=2) predictions match keras3 predict", {
    .keras_skip()
    k <- reticulate::import("keras")
    skip_if(
        is.null(tryCatch(k$layers$AdaptiveMaxPooling1D, error = function(e) {
            NULL
        })),
        "AdaptiveMaxPooling1D not available in this Keras version"
    )
    T_len <- 6L
    C_in <- 4L
    inp <- k$Input(shape = list(T_len, C_in))
    x <- k$layers$AdaptiveMaxPooling1D(output_size = 2L)(inp)
    x <- k$layers$Flatten()(x)
    out <- k$layers$Dense(1L)(x)
    model <- k$Model(inputs = inp, outputs = out)
    model$compile(optimizer = "adam", loss = "mse")

    set.seed(42)
    n_row <- 10L
    x_arr <- array(rnorm(n_row * T_len * C_in), dim = c(n_row, T_len, C_in))
    model$fit(x_arr, rnorm(n_row), epochs = 2L, verbose = 0L)

    x_flat <- matrix(as.numeric(x_arr), nrow = n_row)
    feature_names <- paste0("x", seq_len(ncol(x_flat)))
    df <- as.data.frame(x_flat)
    names(df) <- feature_names
    orb_obj <- orbital(
        model,
        mode = "regression",
        feature_names = feature_names
    )
    expect_true(any(grepl("orbital_adaptivemaxpool1d_", names(orb_obj))))
    expect_equal(
        predict(orb_obj, df)$.pred,
        as.numeric(model$predict(x_arr, verbose = 0L)),
        tolerance = 1e-4
    )
})

# ── EinsumDense ("...b,bc->...c") ─────────────────────────────────────────────

test_that("keras3 EinsumDense (\"...b,bc->...c\") predictions match keras3 predict", {
    .keras_skip()
    k <- reticulate::import("keras")
    inp <- k$Input(shape = list(4L))
    x <- k$layers$EinsumDense(
        equation = "...b,bc->...c",
        output_shape = 6L,
        bias_axes = "c",
        activation = "relu"
    )(inp)
    out <- k$layers$Dense(1L)(x)
    model <- k$Model(inputs = inp, outputs = out)
    model$compile(optimizer = "adam", loss = "mse")

    set.seed(42)
    x_mat <- matrix(rnorm(40L), nrow = 10L, ncol = 4L)
    y_vec <- rnorm(10L)
    model$fit(x_mat, y_vec, epochs = 5L, verbose = 0L)

    feature_names <- paste0("x", 1:4)
    df <- as.data.frame(x_mat)
    names(df) <- feature_names
    orb_obj <- orbital(
        model,
        mode = "regression",
        feature_names = feature_names
    )
    expect_true(any(grepl("orbital_einsumdense_", names(orb_obj))))
    expect_equal(
        predict(orb_obj, df)$.pred,
        as.numeric(model$predict(x_mat, verbose = 0L)),
        tolerance = 1e-5
    )
})
