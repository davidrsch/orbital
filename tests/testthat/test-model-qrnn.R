test_that("as_qrnn_fit() attaches class to a valid qrnn list", {
  skip_if_not_installed("qrnn")

  set.seed(42)
  m_raw <- qrnn::qrnn.fit(
    x = as.matrix(mtcars[, c("cyl", "disp", "hp")]),
    y = as.matrix(mtcars[, "mpg", drop = FALSE]),
    n.hidden = 3L,
    tau = 0.5,
    iter.max = 10,
    n.trials = 1,
    trace = FALSE
  )

  m <- as_qrnn_fit(m_raw)
  expect_s3_class(m, "qrnn_fit")
  expect_true(inherits(m, "list"))
})

test_that("as_qrnn_fit() rejects non-qrnn lists", {
  expect_error(
    as_qrnn_fit(list(a = 1, b = 2)),
    "does not look like a qrnn fit"
  )
})

test_that("as_qrnn_fit() rejects non-lists", {
  expect_error(as_qrnn_fit(42), "requires a list")
})

test_that("orbital.qrnn_fit() regression matches qrnn.predict()", {
  skip_if_not_installed("qrnn")

  set.seed(42)
  x_mat <- as.matrix(mtcars[, c("cyl", "disp", "hp")])
  y_mat <- as.matrix(mtcars[, "mpg", drop = FALSE])

  m_raw <- qrnn::qrnn.fit(
    x = x_mat,
    y = y_mat,
    n.hidden = 3L,
    tau = 0.5,
    iter.max = 50,
    n.trials = 1,
    trace = FALSE
  )
  m <- as_qrnn_fit(m_raw)

  orb_obj <- orbital(m)
  expect_s3_class(orb_obj, "orbital_class")

  preds_orb <- predict(orb_obj, mtcars)
  preds_qrnn <- as.numeric(qrnn::qrnn.predict(x_mat, m_raw))

  expect_named(preds_orb, ".pred")
  expect_type(preds_orb$.pred, "double")
  expect_equal(preds_orb$.pred, preds_qrnn, tolerance = 1e-5)
})

test_that("orbital.qrnn_fit() works with tanh(0.5*x) (qrnn::sigmoid) activation", {
  skip_if_not_installed("qrnn")

  set.seed(7)
  x_mat <- as.matrix(mtcars[, c("wt", "hp")])
  y_mat <- as.matrix(mtcars[, "mpg", drop = FALSE])

  m_raw <- qrnn::qrnn.fit(
    x = x_mat,
    y = y_mat,
    n.hidden = 4L,
    tau = 0.9,
    Th = qrnn::sigmoid,
    iter.max = 30,
    n.trials = 1,
    trace = FALSE
  )
  m <- as_qrnn_fit(m_raw)

  orb_obj <- orbital(m)
  preds_orb <- predict(orb_obj, mtcars)
  preds_ref <- as.numeric(qrnn::qrnn.predict(x_mat, m_raw))

  expect_equal(preds_orb$.pred, preds_ref, tolerance = 1e-5)
})

test_that("orbital.qrnn_fit() works with relu activation", {
  skip_if_not_installed("qrnn")

  set.seed(99)
  x_mat <- as.matrix(mtcars[, c("wt", "hp", "cyl")])
  y_mat <- as.matrix(mtcars[, "mpg", drop = FALSE])

  m_raw <- qrnn::qrnn.fit(
    x = x_mat,
    y = y_mat,
    n.hidden = 5L,
    tau = 0.5,
    Th = qrnn::relu,
    iter.max = 30,
    n.trials = 1,
    trace = FALSE
  )
  m <- as_qrnn_fit(m_raw)

  orb_obj <- orbital(m)
  preds_orb <- predict(orb_obj, mtcars)
  preds_ref <- as.numeric(qrnn::qrnn.predict(x_mat, m_raw))

  expect_equal(preds_orb$.pred, preds_ref, tolerance = 1e-5)
})

test_that("orbital.qrnn_fit() works with logistic (qrnn::logistic) activation", {
  skip_if_not_installed("qrnn")

  set.seed(5)
  x_mat <- as.matrix(mtcars[, c("wt", "disp")])
  y_mat <- as.matrix(mtcars[, "mpg", drop = FALSE])

  m_raw <- qrnn::qrnn.fit(
    x = x_mat,
    y = y_mat,
    n.hidden = 3L,
    tau = 0.5,
    Th = qrnn::logistic,
    iter.max = 30,
    n.trials = 1,
    trace = FALSE
  )
  m <- as_qrnn_fit(m_raw)

  orb_obj <- orbital(m)
  preds_orb <- predict(orb_obj, mtcars)
  preds_ref <- as.numeric(qrnn::qrnn.predict(x_mat, m_raw))

  expect_equal(preds_orb$.pred, preds_ref, tolerance = 1e-5)
})

# A minimal synthetic qrnn_fit-like structure that bypasses qrnn::qrnn.fit
# (so tests run without the qrnn package installed).
# W1: (n_inputs+1) x n_hidden — last row = bias
# W2: (n_hidden+1) x 1 — last row = bias
.make_synthetic_qrnn <- function(act_fn, x_vals = c(1, 2), seed = 42) {
  set.seed(seed)
  n_inputs <- length(x_vals)
  n_hidden <- 3L
  W1 <- matrix(
    rnorm((n_inputs + 1L) * n_hidden),
    nrow = n_inputs + 1L,
    ncol = n_hidden
  )
  W2 <- matrix(rnorm(n_hidden + 1L), nrow = n_hidden + 1L, ncol = 1L)
  list(
    weights = list(list(W1 = W1, W2 = W2)),
    Th = act_fn,
    x.center = setNames(rep(0, n_inputs), paste0("x", seq_len(n_inputs))),
    x.scale = setNames(rep(1, n_inputs), paste0("x", seq_len(n_inputs))),
    y.center = 0,
    y.scale = 1,
    lower = -Inf,
    monotone = NULL,
    additive = FALSE
  )
}

test_that("orbital.qrnn_fit works with softplus activation", {
  skip_if_not_installed("qrnn")
  m <- as_qrnn_fit(.make_synthetic_qrnn(qrnn::softplus))
  orb <- orbital(m)
  tbl <- tibble::tibble(x1 = 1.0, x2 = 2.0)
  result <- predict(orb, tbl)
  expect_true(is.data.frame(result))
  expect_true(is.finite(result[[".pred"]][1L]))
})

test_that("orbital.qrnn_fit works with lrelu activation", {
  skip_if_not_installed("qrnn")
  m <- as_qrnn_fit(.make_synthetic_qrnn(qrnn::lrelu))
  orb <- orbital(m)
  tbl <- tibble::tibble(x1 = 1.0, x2 = 2.0)
  result <- predict(orb, tbl)
  expect_true(is.data.frame(result))
  expect_true(is.finite(result[[".pred"]][1L]))
})

test_that("orbital.qrnn_fit works with elu activation (alpha=1)", {
  skip_if_not_installed("qrnn")
  m <- as_qrnn_fit(.make_synthetic_qrnn(qrnn::elu))
  orb <- orbital(m)
  tbl <- tibble::tibble(x1 = 1.0, x2 = 2.0)
  result <- predict(orb, tbl)
  expect_true(is.data.frame(result))
  expect_true(is.finite(result[[".pred"]][1L]))
})

test_that("orbital.qrnn_fit works with linear activation (no spurious warning)", {
  skip_if_not_installed("qrnn")
  m <- as_qrnn_fit(.make_synthetic_qrnn(qrnn::linear))
  expect_no_warning(
    orbital(m),
    class = "orbital_qrnn_unknown_activation"
  )
  orb <- orbital(m)
  tbl <- tibble::tibble(x1 = 1.0, x2 = 2.0)
  result <- predict(orb, tbl)
  expect_true(is.data.frame(result))
})

test_that("as_qrnn_fit with n.ensemble > 1 aborts in orbital()", {
  base <- .make_synthetic_qrnn(function(x) tanh(0.5 * x)) # qrnn_sigmoid
  base$weights <- c(base$weights, base$weights) # simulate n.ensemble = 2
  m <- structure(base, class = c("qrnn_fit", "list"))
  expect_error(orbital(m), class = "rlang_error")
})

test_that("orbital.qrnn_fit aborts for monotone-constrained models", {
  base <- .make_synthetic_qrnn(function(x) tanh(0.5 * x))
  base$monotone <- 1L
  m <- structure(base, class = c("qrnn_fit", "list"))
  expect_error(orbital(m), class = "rlang_error")
})

test_that("orbital.qrnn_fit aborts for censored models (lower > -Inf)", {
  base <- .make_synthetic_qrnn(function(x) tanh(0.5 * x))
  base$lower <- 0
  m <- structure(base, class = c("qrnn_fit", "list"))
  expect_error(orbital(m), class = "rlang_error")
})

test_that("orbital.qrnn_fit works when scale.y = FALSE (y.center=0, y.scale=1)", {
  # .make_synthetic_qrnn already sets y.center=0 and y.scale=1.
  base <- .make_synthetic_qrnn(function(x) tanh(0.5 * x))
  m <- as_qrnn_fit(base)
  orb <- orbital(m)
  tbl <- tibble::tibble(x1 = 1.0, x2 = 2.0)
  result <- predict(orb, tbl)
  expect_true(is.data.frame(result))
  expect_true(is.finite(result[[".pred"]][1L]))
})
