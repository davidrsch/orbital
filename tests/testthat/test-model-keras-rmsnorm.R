# Integration tests for keras3 RMSNormalization layer (finding R-04 / E6).
# These tests are skipped when required packages are not installed.

test_that("R-04: keras3 RMSNormalization translates and matches keras3 predict", {
    skip_if_no_keras3()
    k <- reticulate::import("keras")

    # Toy model: Dense -> RMSNormalization -> Dense(1). RMSNormalization
    # normalises each row by its root-mean-square value (no mean subtraction)
    # and scales by a learnable gamma vector. This is the minimal sanity
    # check that .k3_rmsnorm() emits a valid per-unit expression block.
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

    feature_names <- paste0("x", seq_len(4L))
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
