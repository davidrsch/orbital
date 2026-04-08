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
