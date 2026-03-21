# Integration tests for H2O DeepLearning MLP engine (agua package)
# All tests are skipped when H2O is not installed or no running H2O server
# is detected, so they are safe to run in CI without H2O.

.h2o_available <- function() {
    if (!rlang::is_installed("h2o") || !rlang::is_installed("agua")) {
        return(FALSE)
    }
    tryCatch(
        {
            h2o::h2o.getConnection()
            TRUE
        },
        error = function(e) FALSE
    )
}

test_that("mlp() h2o engine works with regression", {
    skip_if_not_installed("agua")
    skip_if_not_installed("h2o")
    skip_if_not_installed("parsnip")
    skip_if(!.h2o_available(), "H2O server not available")

    spec <- parsnip::mlp(hidden_units = 4, epochs = 20, engine = "h2o")
    spec <- parsnip::set_mode(spec, "regression")

    set.seed(1)
    fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

    orb_obj <- orbital(fit)
    preds <- predict(orb_obj, mtcars)
    exps <- predict(fit, mtcars)

    expect_named(preds, ".pred")
    expect_type(preds$.pred, "double")

    exps <- as.data.frame(exps)
    rownames(preds) <- NULL
    rownames(exps) <- NULL
    expect_equal(preds, exps, tolerance = 1e-4)
})

test_that("mlp() h2o engine works with binary classification", {
    skip_if_not_installed("agua")
    skip_if_not_installed("h2o")
    skip_if_not_installed("parsnip")
    skip_if(!.h2o_available(), "H2O server not available")

    mtcars$vs <- factor(mtcars$vs)
    spec <- parsnip::mlp(hidden_units = 4, epochs = 20, engine = "h2o")
    spec <- parsnip::set_mode(spec, "classification")

    set.seed(1)
    fit <- parsnip::fit(spec, vs ~ disp + wt + hp, mtcars)

    orb_obj <- orbital(fit, type = "class")
    preds <- predict(orb_obj, mtcars)
    exps <- predict(fit, mtcars)

    expect_named(preds, ".pred_class")
    expect_identical(preds$.pred_class, as.character(exps$.pred_class))
})

test_that("mlp() h2o engine works with binary prob", {
    skip_if_not_installed("agua")
    skip_if_not_installed("h2o")
    skip_if_not_installed("parsnip")
    skip_if(!.h2o_available(), "H2O server not available")

    mtcars$vs <- factor(mtcars$vs)
    spec <- parsnip::mlp(hidden_units = 4, epochs = 20, engine = "h2o")
    spec <- parsnip::set_mode(spec, "classification")

    set.seed(1)
    fit <- parsnip::fit(spec, vs ~ disp + wt + hp, mtcars)

    orb_obj <- orbital(fit, type = "prob")
    preds <- predict(orb_obj, mtcars)
    exps <- predict(fit, mtcars, type = "prob")

    lvls <- levels(mtcars$vs)
    expect_named(preds, paste0(".pred_", lvls))
    expect_equal(preds, as.data.frame(exps), tolerance = 1e-4)
})

test_that("mlp() h2o with tanh activation works", {
    skip_if_not_installed("agua")
    skip_if_not_installed("h2o")
    skip_if_not_installed("parsnip")
    skip_if(!.h2o_available(), "H2O server not available")

    spec <- parsnip::mlp(hidden_units = 4, epochs = 10, engine = "h2o")
    spec <- parsnip::set_mode(spec, "regression")
    spec <- parsnip::set_engine(spec, "h2o", activation = "Tanh")

    set.seed(1)
    fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

    orb_obj <- orbital(fit)
    preds <- predict(orb_obj, mtcars)
    expect_named(preds, ".pred")
})

test_that("mlp() h2o Maxout activation raises informative error", {
    skip_if_not_installed("agua")
    skip_if_not_installed("h2o")
    skip_if_not_installed("parsnip")
    skip_if(!.h2o_available(), "H2O server not available")

    spec <- parsnip::mlp(hidden_units = 4, epochs = 5, engine = "h2o")
    spec <- parsnip::set_mode(spec, "regression")
    spec <- parsnip::set_engine(spec, "h2o", activation = "Maxout")

    set.seed(1)
    fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

    expect_error(orbital(fit), "Maxout")
})
