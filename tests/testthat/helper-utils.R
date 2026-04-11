## Skip guard for keras3 + reticulate tests
skip_if_no_keras3 <- function() {
  skip_if_not_installed("keras3")
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_available(initialize = FALSE),
    "Python not available"
  )
}

## Alias used in newer test files (equivalent to skip_if_no_keras3)
.keras_skip <- function() {
  skip_if_no_keras3()
}

## For sparklyr testing

testthat_tbl <- function(name, data = NULL, repartition = 0L) {
  sc <- testthat_spark_connection()

  tbl <- tryCatch(dplyr::tbl(sc, name), error = identity)
  if (inherits(tbl, "error")) {
    if (is.null(data)) {
      data <- eval(as.name(name), envir = parent.frame())
    }
    tbl <- dplyr::copy_to(sc, data, name = name, repartition = repartition)
  }

  tbl
}

make_seq_test_data <- function(n_row, T_len, C_in, seed = 42L) {
  set.seed(seed)
  x_flat <- matrix(
    rnorm(n_row * T_len * C_in),
    nrow = n_row,
    ncol = T_len * C_in
  )
  x_3d <- array(x_flat, dim = c(n_row, T_len, C_in))
  feature_names <- paste0("x", seq_len(T_len * C_in))
  df <- as.data.frame(x_flat)
  names(df) <- feature_names

  list(
    x_flat = x_flat,
    x_3d = x_3d,
    feature_names = feature_names,
    df = df,
    y_vec = rnorm(n_row)
  )
}

run_orbital_and_compare <- function(
  model,
  new_data,
  expected_input,
  feature_names,
  tol = 1e-4
) {
  orb_obj <- orbital(model, mode = "regression", feature_names = feature_names)
  preds_orb <- predict(orb_obj, new_data)$.pred
  preds_ref <- as.numeric(model$predict(expected_input, verbose = 0L))
  testthat::expect_equal(preds_orb, preds_ref, tolerance = tol)

  invisible(list(orbital = preds_orb, reference = preds_ref))
}
