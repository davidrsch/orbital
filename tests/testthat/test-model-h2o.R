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

# ---------------------------------------------------------------------------
# Mock-based unit tests (no live H2O server required)
# ---------------------------------------------------------------------------

# Minimal S4 mock mirroring the slots that orbital reads from an
# H2ODeepLearningModel. Defined only for the scope of these tests.
setClass(
  "MockH2OModel",
  representation(
    parameters = "list",
    allparameters = "list",
    model = "list"
  )
)

make_mock_h2o <- function(
  response_column = "Species",
  output_names = c(
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "Species"
  ),
  domains = list(
    NULL,
    NULL,
    NULL,
    NULL,
    c("setosa", "versicolor", "virginica")
  ),
  parameters = list()
) {
  methods::new(
    "MockH2OModel",
    parameters = c(parameters, list(response_column = response_column)),
    allparameters = list(response_column = response_column),
    model = list(output = list(names = output_names, domains = domains))
  )
}

test_that("h2o_response_levels() returns class levels in H2O order", {
  mock <- make_mock_h2o()
  expect_identical(
    h2o_response_levels(mock),
    c("setosa", "versicolor", "virginica")
  )
})

test_that("h2o_response_levels() returns NULL when metadata is missing", {
  mock_no_names <- methods::new(
    "MockH2OModel",
    parameters = list(response_column = "y"),
    allparameters = list(response_column = "y"),
    model = list(output = list(domains = list(c("a", "b"))))
  )
  expect_null(h2o_response_levels(mock_no_names))

  mock_no_match <- make_mock_h2o(response_column = "not_in_names")
  expect_null(h2o_response_levels(mock_no_match))

  mock_no_domain <- make_mock_h2o(
    domains = list(NULL, NULL, NULL, NULL, NULL)
  )
  expect_null(h2o_response_levels(mock_no_domain))
})

test_that("H2O Maxout without maxout_size aborts (no silent fallback)", {
  # Regression test for finding H-01: previously the absence of maxout_size
  # produced a warning + silent fallback to 2, which could mispredict.
  mock <- methods::new(
    "MockH2OModel",
    parameters = list(activation = "Maxout"), # maxout_size deliberately absent
    allparameters = list(),
    model = list(output = list(names = character(), domains = list()))
  )
  expect_error(
    orbital_h2o_dl_impl(
      mock,
      mode = "regression",
      type = "numeric",
      lvl = NULL,
      prefix = ".pred"
    ),
    regexp = "maxout_size"
  )
})

test_that("H2O input_norm_sub malformed aborts", {
  # Regression test for finding H-03.
  mock <- methods::new(
    "MockH2OModel",
    parameters = list(activation = "Rectifier"),
    allparameters = list(),
    model = list(
      input_norm_sub = c(1, NA, 3),
      input_norm_mul = c(1, 2, 3),
      output = list(names = character(), domains = list())
    )
  )
  expect_error(
    orbital_h2o_dl_impl(
      mock,
      mode = "regression",
      type = "numeric",
      lvl = NULL,
      prefix = ".pred"
    ),
    regexp = "input_norm_sub"
  )
})
