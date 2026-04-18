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
  orb <- orbital(fit, mode = "regression")
  preds <- predict(orb, mtcars)
  exps <- as.data.frame(predict(fit, mtcars))
  names(exps) <- ".pred"
  rownames(preds) <- NULL
  rownames(exps) <- NULL
  expect_equal(preds, exps, tolerance = 1e-6)
})

test_that("nnet classification refuses linout = TRUE", {
  skip_if_not_installed("nnet")

  df <- mtcars
  df$vs <- factor(df$vs)
  set.seed(1)
  fit <- nnet::nnet(
    vs ~ disp + wt + hp,
    data = df,
    size = 3,
    trace = FALSE,
    linout = TRUE
  )
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
