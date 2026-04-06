.kerasnip_skip <- function() {
    skip_if_not_installed("kerasnip")
    skip_if_not_installed("keras3")
    skip_if_not_installed("reticulate")
    skip_if_not(
        reticulate::py_available(initialize = FALSE),
        "Python not available"
    )
}

# Shared layer-block builders used across tests ---------------------------

.ks_input <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
}

.ks_dense <- function(model, units = 8L) {
    model |> keras3::layer_dense(units = units, activation = "relu")
}

.ks_reg_output <- function(model) {
    model |> keras3::layer_dense(units = 1L)
}

.ks_cls_output <- function(model, num_classes) {
    model |> keras3::layer_dense(units = num_classes, activation = "sigmoid")
}


# ── Regression tests ──────────────────────────────────────────────────────────

test_that("kerasnip sequential regression: orbital() runs without error", {
    .kerasnip_skip()

    kerasnip::create_keras_sequential_spec(
        model_name = "ks_smoke_reg",
        layer_blocks = list(
            input = .ks_input,
            hidden = .ks_dense,
            output = .ks_reg_output
        ),
        mode = "regression"
    )
    on.exit(kerasnip::remove_keras_spec("ks_smoke_reg"), add = TRUE)

    spec <- ks_smoke_reg(hidden_units = 8L, fit_epochs = 3L) |>
        parsnip::set_engine("keras")

    set.seed(42)
    fit_obj <- parsnip::fit(spec, mpg ~ disp + wt + hp, data = mtcars)

    expect_no_error(orbital(fit_obj))
})

test_that("kerasnip sequential regression predictions match kerasnip predict()", {
    .kerasnip_skip()

    kerasnip::create_keras_sequential_spec(
        model_name = "ks_acc_reg",
        layer_blocks = list(
            input = .ks_input,
            hidden = .ks_dense,
            output = .ks_reg_output
        ),
        mode = "regression"
    )
    on.exit(kerasnip::remove_keras_spec("ks_acc_reg"), add = TRUE)

    spec <- ks_acc_reg(hidden_units = 8L, fit_epochs = 5L) |>
        parsnip::set_engine("keras")

    set.seed(42)
    fit_obj <- parsnip::fit(spec, mpg ~ disp + wt + hp, data = mtcars)

    orb_obj <- orbital(fit_obj)
    preds_orb <- predict(orb_obj, mtcars)
    preds_ks <- predict(fit_obj, new_data = mtcars)

    expect_named(preds_orb, ".pred")

    rownames(preds_orb) <- NULL
    preds_ref <- as.data.frame(preds_ks)
    rownames(preds_ref) <- NULL

    expect_equal(preds_orb, preds_ref, tolerance = 1e-4)
})

test_that("kerasnip sequential multi-layer regression works", {
    .kerasnip_skip()

    kerasnip::create_keras_sequential_spec(
        model_name = "ks_deep_reg",
        layer_blocks = list(
            input = .ks_input,
            hidden1 = .ks_dense,
            hidden2 = .ks_dense,
            output = .ks_reg_output
        ),
        mode = "regression"
    )
    on.exit(kerasnip::remove_keras_spec("ks_deep_reg"), add = TRUE)

    spec <- ks_deep_reg(
        hidden1_units = 8L,
        hidden2_units = 4L,
        fit_epochs = 5L
    ) |>
        parsnip::set_engine("keras")

    set.seed(42)
    fit_obj <- parsnip::fit(spec, mpg ~ disp + wt + hp, data = mtcars)

    orb_obj <- orbital(fit_obj)
    preds_orb <- predict(orb_obj, mtcars)
    preds_ks <- predict(fit_obj, new_data = mtcars)

    rownames(preds_orb) <- NULL
    preds_ref <- as.data.frame(preds_ks)
    rownames(preds_ref) <- NULL

    expect_equal(preds_orb, preds_ref, tolerance = 1e-4)
})


# ── Classification tests ──────────────────────────────────────────────────────

test_that("kerasnip sequential binary classification (class type) works", {
    .kerasnip_skip()

    kerasnip::create_keras_sequential_spec(
        model_name = "ks_bin_class",
        layer_blocks = list(
            input = .ks_input,
            hidden = .ks_dense,
            output = .ks_cls_output
        ),
        mode = "classification"
    )
    on.exit(kerasnip::remove_keras_spec("ks_bin_class"), add = TRUE)

    mtcars_cls <- mtcars
    mtcars_cls$vs <- factor(mtcars_cls$vs)

    spec <- ks_bin_class(hidden_units = 8L, fit_epochs = 5L) |>
        parsnip::set_engine("keras")

    set.seed(42)
    fit_obj <- parsnip::fit(spec, vs ~ disp + wt + hp, data = mtcars_cls)

    orb_obj <- orbital(fit_obj, type = "class")
    preds_orb <- predict(orb_obj, mtcars_cls)
    preds_ks <- predict(fit_obj, new_data = mtcars_cls)

    expect_named(preds_orb, ".pred_class")
    expect_identical(
        as.character(preds_orb$.pred_class),
        as.character(preds_ks$.pred_class)
    )
})

test_that("kerasnip sequential binary classification (prob type) predictions match", {
    .kerasnip_skip()

    kerasnip::create_keras_sequential_spec(
        model_name = "ks_bin_prob",
        layer_blocks = list(
            input = .ks_input,
            hidden = .ks_dense,
            output = .ks_cls_output
        ),
        mode = "classification"
    )
    on.exit(kerasnip::remove_keras_spec("ks_bin_prob"), add = TRUE)

    mtcars_cls <- mtcars
    mtcars_cls$vs <- factor(mtcars_cls$vs)

    spec <- ks_bin_prob(hidden_units = 8L, fit_epochs = 5L) |>
        parsnip::set_engine("keras")

    set.seed(42)
    fit_obj <- parsnip::fit(spec, vs ~ disp + wt + hp, data = mtcars_cls)

    orb_obj <- orbital(fit_obj, type = "prob")
    preds_orb <- predict(orb_obj, mtcars_cls)
    preds_ks <- predict(fit_obj, new_data = mtcars_cls, type = "prob")

    lvls <- levels(mtcars_cls$vs)
    expect_named(preds_orb, paste0(".pred_", lvls))

    rownames(preds_orb) <- NULL
    preds_ref <- as.data.frame(preds_ks)
    rownames(preds_ref) <- NULL

    expect_equal(preds_orb, preds_ref, tolerance = 1e-4)
})

test_that("kerasnip activation variety (tanh, sigmoid) works for regression", {
    .kerasnip_skip()

    make_spec <- function(act) {
        kerasnip::create_keras_sequential_spec(
            model_name = paste0("ks_act_", act),
            layer_blocks = list(
                input = .ks_input,
                hidden = function(model, units = 8L) {
                    model |>
                        keras3::layer_dense(units = units, activation = act)
                },
                output = .ks_reg_output
            ),
            mode = "regression"
        )
        spec_fn <- get(paste0("ks_act_", act))
        spec_fn(hidden_units = 8L, fit_epochs = 3L) |>
            parsnip::set_engine("keras")
    }

    on.exit(
        {
            for (act in c("tanh", "sigmoid")) {
                tryCatch(
                    kerasnip::remove_keras_spec(paste0("ks_act_", act)),
                    error = function(e) NULL
                )
            }
        },
        add = TRUE
    )

    set.seed(42)
    df <- data.frame(
        x1 = rnorm(30),
        x2 = rnorm(30),
        x3 = rnorm(30),
        y = rnorm(30)
    )

    for (act in c("tanh", "sigmoid")) {
        spec <- make_spec(act)
        fit_obj <- parsnip::fit(spec, y ~ x1 + x2 + x3, data = df)
        orb_obj <- orbital(fit_obj)
        preds_orb <- predict(orb_obj, df)
        preds_ks <- predict(fit_obj, new_data = df)

        rownames(preds_orb) <- NULL
        preds_ref <- as.data.frame(preds_ks)
        rownames(preds_ref) <- NULL

        expect_equal(
            preds_orb,
            preds_ref,
            tolerance = 1e-4,
            info = paste("activation:", act)
        )
    }
})
