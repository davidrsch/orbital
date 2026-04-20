test_that("nnet regression refuses linout = FALSE (H5 / M12)", {
  skip_if_not_installed("nnet")

  # Direct nnet::nnet() fit with default linout = FALSE, which means nnet
  # internally applies a logistic squash. Emitting the raw pre-activation
  # would silently produce wrong predictions.
  set.seed(1)
  fit <- nnet::nnet(
    mpg ~ disp + wt + hp,
    data = mtcars,
    size = 3,
    trace = FALSE,
    linout = FALSE
  )
  expect_false(isTRUE(fit$linout))
  expect_error(
    orbital(fit, mode = "regression"),
    "linout = TRUE"
  )
})

test_that("nnet regression accepts linout = TRUE", {
  skip_if_not_installed("nnet")

  set.seed(1)
  fit <- nnet::nnet(
    mpg ~ disp + wt + hp,
    data = mtcars,
    size = 3,
    trace = FALSE,
    linout = TRUE
  )
  # orbital() on a bare nnet fit returns the language expression directly;
  # the parity-with-predict() check is covered by the parsnip-based test
  # in test-model-nnet.R. Here we only verify the linout = TRUE guard
  # accepts the fit without erroring.
  expect_no_error(orbital(fit, mode = "regression"))
})

test_that("nnet classification refuses linout = TRUE", {
  skip_if_not_installed("nnet")

  # nnet itself rejects linout = TRUE on a factor outcome (entropy = TRUE
  # is set automatically by nnet.formula for factors, and
  # `entropy && linout` is invalid). Fit normally, then mutate the fit
  # object to the unsupported combination to exercise orbital's guard.
  df <- mtcars
  df$vs <- factor(df$vs)
  set.seed(1)
  fit <- nnet::nnet(
    vs ~ disp + wt + hp,
    data = df,
    size = 3,
    trace = FALSE
  )
  fit$linout <- TRUE
  expect_error(
    orbital(fit, mode = "classification", lvl = levels(df$vs)),
    "linout = FALSE"
  )
})

test_that("nnet refuses censored = TRUE", {
  skip_if_not_installed("nnet")

  # Fake a censored fit so we can exercise the validation branch without a
  # real multinomial dataset.
  set.seed(1)
  fit <- nnet::nnet(
    mpg ~ disp + wt + hp,
    data = mtcars,
    size = 3,
    trace = FALSE,
    linout = TRUE
  )
  fit$censored <- TRUE
  expect_error(
    orbital(fit, mode = "regression"),
    "censored = TRUE"
  )
})

# ---------------------------------------------------------------------------
# Phase 9 additions (E2, E3, E9): N-01 weight-length, N-02 bag_mlp, N-03 softmax
# ---------------------------------------------------------------------------

test_that("N-01: nnet fit with truncated weight vector aborts with length mismatch", {
  skip_if_not_installed("nnet")

  set.seed(1)
  fit <- nnet::nnet(
    mpg ~ disp + wt + hp,
    data = mtcars,
    size = 3,
    trace = FALSE,
    linout = TRUE
  )
  # Mutate wts to a wrong length: drop the last weight element.
  fit$wts <- fit$wts[-length(fit$wts)]
  expect_error(
    orbital(fit, mode = "regression"),
    regexp = "[Uu]nexpected|weight"
  )
})

test_that("N-02: bag_mlp objects are rejected with a pointer to extract_fit_engine", {
  # parsnip::bag_mlp wraps multiple nnet fits; orbital does not yet dispatch
  # on the ensemble object directly and refuses with a message that points
  # the user to the per-constituent workaround.
  # orbital.nnet() is the guard site; exercise it directly rather than
  # routing through orbital.model_fit() which first inspects `spec$mode`.
  stub <- structure(list(), class = c("bag_mlp"))
  expect_error(
    orbital:::orbital.nnet(stub),
    regexp = "bag_mlp|extract_fit_engine"
  )
})

test_that("N-03: nnet binary classification aborts when softmax = TRUE", {
  skip_if_not_installed("nnet")

  df <- mtcars
  df$vs <- factor(df$vs)
  set.seed(1)
  fit <- nnet::nnet(
    vs ~ disp + wt + hp,
    data = df,
    size = 3,
    trace = FALSE,
    linout = FALSE
  )
  # Binary fit has n_out == 1; forcing softmax = TRUE is not a supported
  # combination because nnet emits a single sigmoid unit for binary.
  fit$softmax <- TRUE
  expect_error(
    orbital(fit, mode = "classification", lvl = levels(df$vs)),
    regexp = "softmax"
  )
})

test_that("N-03: nnet multiclass classification aborts when softmax = FALSE", {
  skip_if_not_installed("nnet")

  set.seed(1)
  fit <- nnet::nnet(
    Species ~ Sepal.Length + Sepal.Width + Petal.Length,
    data = iris,
    size = 3,
    trace = FALSE,
    linout = FALSE
  )
  # Multi-class fits normally set softmax = TRUE; forcing it FALSE means the
  # outputs are independent logistic units instead of a softmax simplex.
  fit$softmax <- FALSE
  expect_error(
    orbital(fit, mode = "classification", lvl = levels(iris$Species)),
    regexp = "softmax"
  )
})
