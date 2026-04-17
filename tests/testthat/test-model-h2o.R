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

test_that("mlp() h2o with TanhWithDropout activation works (inference = Tanh)", {
  skip_if_not_installed("agua")
  skip_if_not_installed("h2o")
  skip_if_not_installed("parsnip")
  skip_if(!.h2o_available(), "H2O server not available")

  spec <- parsnip::mlp(hidden_units = 4, epochs = 10, engine = "h2o")
  spec <- parsnip::set_mode(spec, "regression")
  spec <- parsnip::set_engine(spec, "h2o", activation = "TanhWithDropout")

  set.seed(1)
  fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit)
  preds <- predict(orb_obj, mtcars)
  exps <- predict(fit, mtcars)

  expect_named(preds, ".pred")
  exps <- as.data.frame(exps)
  rownames(preds) <- NULL
  rownames(exps) <- NULL

  expect_equal(preds, exps, tolerance = 1e-4)
})

test_that("H2O dropout activation aliases map to the same inference expressions offline", {
  expect_identical(
    orbital:::activation_expr("TanhWithDropout", "z"),
    orbital:::activation_expr("Tanh", "z")
  )
  expect_identical(
    orbital:::activation_expr("RectifierWithDropout", "z"),
    orbital:::activation_expr("Rectifier", "z")
  )
  # Maxout/MaxoutWithDropout are intercepted in model-h2o.R via the is_maxout
  # flag before activation_expr() is called; they are not passed to
  # activation_expr() during inference. End-to-end Maxout coverage is provided
  # by the mlp() integration tests below.
})

test_that("mlp() h2o Maxout activation is translated correctly", {
  skip_if_not_installed("agua")
  skip_if_not_installed("h2o")
  skip_if_not_installed("parsnip")
  skip_if(!.h2o_available(), "H2O server not available")

  spec <- parsnip::mlp(hidden_units = 4, epochs = 10, engine = "h2o")
  spec <- parsnip::set_mode(spec, "regression")
  spec <- parsnip::set_engine(spec, "h2o", activation = "Maxout")

  set.seed(1)
  fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit)
  preds_orb <- predict(orb_obj, mtcars)
  preds_fit <- predict(fit, mtcars)
  expect_equal(preds_orb$.pred, preds_fit$.pred, tolerance = 1e-5)
})

test_that("mlp() h2o engine works with multiclass probability", {
  skip_if_not_installed("agua")
  skip_if_not_installed("h2o")
  skip_if_not_installed("parsnip")
  skip_if(!.h2o_available(), "H2O server not available")

  spec <- parsnip::mlp(hidden_units = 4, epochs = 20, engine = "h2o")
  spec <- parsnip::set_mode(spec, "classification")

  set.seed(1)
  fit <- parsnip::fit(
    spec,
    Species ~ Sepal.Length + Sepal.Width + Petal.Length,
    iris
  )

  orb_obj <- orbital(fit, type = "prob")
  preds <- predict(orb_obj, iris)
  exps <- predict(fit, iris, type = "prob")

  lvls <- levels(iris$Species)
  expect_named(preds, paste0(".pred_", lvls))

  exps <- as.data.frame(exps)
  rownames(preds) <- NULL
  rownames(exps) <- NULL

  expect_equal(preds, exps, tolerance = 1e-4)
})

test_that("mlp() h2o engine works with multiclass class", {
  skip_if_not_installed("agua")
  skip_if_not_installed("h2o")
  skip_if_not_installed("parsnip")
  skip_if(!.h2o_available(), "H2O server not available")

  spec <- parsnip::mlp(hidden_units = 4, epochs = 20, engine = "h2o")
  spec <- parsnip::set_mode(spec, "classification")

  set.seed(1)
  fit <- parsnip::fit(
    spec,
    Species ~ Sepal.Length + Sepal.Width + Petal.Length,
    iris
  )

  orb_obj <- orbital(fit, type = "class")
  preds <- predict(orb_obj, iris)
  exps <- predict(fit, iris)

  expect_named(preds, ".pred_class")
  expect_identical(preds$.pred_class, as.character(exps$.pred_class))
})

test_that("mlp() h2o RectifierWithDropout activation works (inference = Rectifier)", {
  skip_if_not_installed("agua")
  skip_if_not_installed("h2o")
  skip_if_not_installed("parsnip")
  skip_if(!.h2o_available(), "H2O server not available")

  spec <- parsnip::mlp(hidden_units = 4, epochs = 10, engine = "h2o")
  spec <- parsnip::set_mode(spec, "regression")
  spec <- parsnip::set_engine(spec, "h2o", activation = "RectifierWithDropout")

  set.seed(1)
  fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit)
  preds <- predict(orb_obj, mtcars)
  exps <- predict(fit, mtcars)

  expect_named(preds, ".pred")
  exps <- as.data.frame(exps)
  rownames(preds) <- NULL
  rownames(exps) <- NULL

  expect_equal(preds, exps, tolerance = 1e-4)
})

test_that("mlp() h2o standardize=FALSE passes raw predictors without normalization", {
  skip_if_not_installed("agua")
  skip_if_not_installed("h2o")
  skip_if_not_installed("parsnip")
  skip_if(!.h2o_available(), "H2O server not available")

  spec <- parsnip::mlp(hidden_units = 4, epochs = 20, engine = "h2o")
  spec <- parsnip::set_mode(spec, "regression")
  spec <- parsnip::set_engine(spec, "h2o", standardize = FALSE)

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

test_that("mlp() h2o Sigmoid activation works", {
  skip_if_not_installed("agua")
  skip_if_not_installed("h2o")
  skip_if_not_installed("parsnip")
  skip_if(!.h2o_available(), "H2O server not available")

  spec <- parsnip::mlp(hidden_units = 4, epochs = 10, engine = "h2o")
  spec <- parsnip::set_mode(spec, "regression")
  spec <- parsnip::set_engine(spec, "h2o", activation = "Sigmoid")

  set.seed(1)
  fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit)
  preds <- predict(orb_obj, mtcars)
  exps <- predict(fit, mtcars)

  expect_named(preds, ".pred")
  exps <- as.data.frame(exps)
  rownames(preds) <- NULL
  rownames(exps) <- NULL
  expect_equal(preds, exps, tolerance = 1e-4)
})

test_that("mlp() h2o MaxoutWithDropout activation is translated correctly (inference = Maxout)", {
  skip_if_not_installed("agua")
  skip_if_not_installed("h2o")
  skip_if_not_installed("parsnip")
  skip_if(!.h2o_available(), "H2O server not available")

  spec <- parsnip::mlp(hidden_units = 4, epochs = 10, engine = "h2o")
  spec <- parsnip::set_mode(spec, "regression")
  spec <- parsnip::set_engine(spec, "h2o", activation = "MaxoutWithDropout")

  set.seed(1)
  fit <- parsnip::fit(spec, mpg ~ disp + wt + hp, mtcars)

  orb_obj <- orbital(fit)
  preds_orb <- predict(orb_obj, mtcars)
  preds_fit <- predict(fit, mtcars)
  expect_equal(preds_orb$.pred, preds_fit$.pred, tolerance = 1e-5)
})

# NOTE: A guard was added to orbital_h2o_dl_impl() that errors when
# length(lvl) != n_out (multiclass output size mismatch). Testing this guard
# requires a live H2O cluster so no automated test is included here.
