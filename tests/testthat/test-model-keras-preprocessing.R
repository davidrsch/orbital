# Unit tests for the Normalization and CategoryEncoding handler functions.
# These tests call the internal handler functions directly using mocked layer
# objects (R environments), so they do not require Keras or Python.

# ── Helper: build a minimal fake Keras-like layer object ─────────────────────

.fake_layer <- function(...) {
    e <- new.env(parent = emptyenv())
    args <- list(...)
    for (nm in names(args)) {
        assign(nm, args[[nm]], envir = e)
    }
    e
}

# ── Group 4: Normalization layer ──────────────────────────────────────────────

test_that(".k3_normalization produces one expression per input feature", {
    n_feat <- 3L
    mn <- c(1.0, 2.0, 3.0)
    vr <- c(0.25, 0.5, 1.0)
    eps <- 1e-3

    l <- .fake_layer(
        get_weights = function() list(mn, vr),
        epsilon = eps
    )

    lname <- "normalization_0"
    in_exprs <- paste0("x", seq_len(n_feat))
    topo_map <- list(normalization_0 = "input_0")
    expr_reg <- new.env(parent = emptyenv())
    assign("input_0", in_exprs, envir = expr_reg)
    state <- new.env(parent = emptyenv())
    state$all_exprs <- list()

    orbital:::.k3_normalization(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map = NULL,
        output_layer_names = NULL,
        last_dense = NULL
    )

    result <- state$all_exprs[[lname]]

    # One expression per input feature
    expect_length(result, n_feat)
    expect_true(all(is.character(result)))

    # Each expression follows the (x - mean) / sqrt(var + eps) shape
    expect_true(all(grepl("/", result, fixed = TRUE)))
    expect_true(all(grepl("sqrt(", result, fixed = TRUE)))

    # The adapt_mean (mn) values must appear in the corresponding expression
    expect_true(grepl("1", result[[1L]]))
    expect_true(grepl("2", result[[2L]]))
    expect_true(grepl("3", result[[3L]]))
})

test_that(".k3_normalization output length matches number of input features for various sizes", {
    for (n_feat in c(1L, 4L, 10L)) {
        mn <- seq_len(n_feat) * 0.5
        vr <- rep(0.25, n_feat)

        l <- .fake_layer(
            get_weights = function() list(mn, vr),
            epsilon = 1e-3
        )

        lname <- "normalization_0"
        in_exprs <- paste0("x", seq_len(n_feat))
        topo_map <- list(normalization_0 = "input_0")
        expr_reg <- new.env(parent = emptyenv())
        assign("input_0", in_exprs, envir = expr_reg)
        state <- new.env(parent = emptyenv())
        state$all_exprs <- list()

        orbital:::.k3_normalization(
            l,
            lname,
            topo_map,
            expr_reg,
            state,
            weight_map = NULL,
            output_layer_names = NULL,
            last_dense = NULL
        )

        expect_length(state$all_exprs[[lname]], n_feat)
    }
})

# ── Group 5: CategoryEncoding — one_hot mode ──────────────────────────────────

test_that(".k3_categoryencoding (one_hot, num_tokens=3) returns 3 as.integer expressions", {
    num_tokens <- 3L

    l <- .fake_layer(
        get_config = function() {
            list(num_tokens = num_tokens, output_mode = "one_hot")
        }
    )

    lname <- "category_encoding_0"
    in_exprs <- "x1"
    topo_map <- list(category_encoding_0 = "input_0")
    expr_reg <- new.env(parent = emptyenv())
    assign("input_0", in_exprs, envir = expr_reg)
    state <- new.env(parent = emptyenv())
    state$all_exprs <- list()

    orbital:::.k3_categoryencoding(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map = NULL,
        output_layer_names = NULL,
        last_dense = NULL
    )

    result <- state$all_exprs[[lname]]

    # num_tokens output expressions
    expect_length(result, num_tokens)
    expect_true(all(is.character(result)))

    # Each expression uses as.integer() and the correct integer token index
    expect_true(all(grepl("as.integer", result, fixed = TRUE)))
    expect_true(grepl("== 0L", result[[1L]], fixed = TRUE))
    expect_true(grepl("== 1L", result[[2L]], fixed = TRUE))
    expect_true(grepl("== 2L", result[[3L]], fixed = TRUE))
})

test_that(".k3_categoryencoding (multi_hot, num_tokens=4) returns 4 as.integer expressions", {
    num_tokens <- 4L

    l <- .fake_layer(
        get_config = function() {
            list(num_tokens = num_tokens, output_mode = "multi_hot")
        }
    )

    lname <- "category_encoding_0"
    in_exprs <- "x1"
    topo_map <- list(category_encoding_0 = "input_0")
    expr_reg <- new.env(parent = emptyenv())
    assign("input_0", in_exprs, envir = expr_reg)
    state <- new.env(parent = emptyenv())
    state$all_exprs <- list()

    orbital:::.k3_categoryencoding(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map = NULL,
        output_layer_names = NULL,
        last_dense = NULL
    )

    result <- state$all_exprs[[lname]]

    expect_length(result, num_tokens)
    expect_true(all(grepl("as.integer", result, fixed = TRUE)))
})

# ── Group 6: CategoryEncoding — count mode raises error ──────────────────────

test_that(".k3_categoryencoding raises an error for output_mode='count'", {
    l <- .fake_layer(
        get_config = function() {
            list(num_tokens = 3L, output_mode = "count")
        }
    )

    lname <- "category_encoding_0"
    in_exprs <- "x1"
    topo_map <- list(category_encoding_0 = "input_0")
    expr_reg <- new.env(parent = emptyenv())
    assign("input_0", in_exprs, envir = expr_reg)
    state <- new.env(parent = emptyenv())
    state$all_exprs <- list()

    expect_error(
        orbital:::.k3_categoryencoding(
            l,
            lname,
            topo_map,
            expr_reg,
            state,
            weight_map = NULL,
            output_layer_names = NULL,
            last_dense = NULL
        ),
        "count"
    )
})
