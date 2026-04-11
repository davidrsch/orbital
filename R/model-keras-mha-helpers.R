# Shared helper functions for Keras MultiHeadAttention / GroupedQueryAttention.
# These helpers centralize 3-D kernel layout resolution and 2-D bias accessors
# used by `model-keras-mha-standard.R` and `model-keras-mha-gqa.R`.

.mha_resolve_3d_kernel <- function(
    k,
    d2,
    d3,
    C_in,
    lname,
    name,
    layer_abbr = "MHA",
    ambiguity_hint = NULL
) {
    d <- dim(k)
    if (length(d) != 3L) {
        cli::cli_abort(
            "{layer_abbr} {.val {lname}}: {name} kernel must be 3-D, got {length(d)}D."
        )
    }
    if (d[1L] != C_in) {
        cli::cli_abort(
            "{layer_abbr} {.val {lname}}: {name} kernel dim1={d[1L]} does not match C_in={C_in}."
        )
    }
    if (d2 == d3) {
        hint <- ambiguity_hint %||%
            paste0(
                "Use a configuration where the projected head dimension differs from the head/group count to avoid this ambiguity."
            )
        cli::cli_abort(
            c(
                "{layer_abbr} {.val {lname}}: {name} kernel dims 2 and 3 are equal ({d2}), so the 3-D kernel layout is ambiguous.",
                "i" = "orbital cannot reliably infer whether the layout is ({C_in}, {d2}, {d3}) or its transpose.",
                "i" = hint
            ),
            class = "orbital_mha_kernel_ambiguous"
        )
    }
    if (d[2L] == d2 && d[3L] == d3) {
        return(function(c_i, i2, i3) k[c_i, i2, i3])
    }
    if (d[2L] == d3 && d[3L] == d2) {
        return(function(c_i, i2, i3) k[c_i, i3, i2])
    }

    cli::cli_abort(
        "{layer_abbr} {.val {lname}}: {name} kernel dims ({paste(d, collapse = 'x')}) cannot be reconciled with the expected ({C_in}, {d2}, {d3}) layout or its transpose."
    )
}

.mha_bias_accessor2 <- function(b, d2, d3) {
    if (is.null(b)) {
        return(function(i2, i3) "0")
    }

    di <- dim(b)
    if (!is.null(di) && length(di) == 2L && di[1L] == d2 && di[2L] == d3) {
        return(function(i2, i3) format_numeric(b[i2, i3]))
    }
    if (!is.null(di) && length(di) == 2L && di[1L] == d3 && di[2L] == d2) {
        return(function(i2, i3) format_numeric(b[i3, i2]))
    }

    function(i2, i3) "0"
}
