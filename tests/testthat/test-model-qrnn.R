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
