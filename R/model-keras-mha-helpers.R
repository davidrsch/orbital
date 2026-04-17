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

# Extract sequence shape variables (C_in, T_q, T_kv, C_out) shared by
# both .k3_multiheadattention() and .k3_groupedqueryattention().
.mha_extract_sequence_shapes <- function(l, q_exprs, kv_exprs) {
  in_shape <- tryCatch(
    as.integer(unlist(l$input_shape[[1L]])),
    error = function(e) NULL
  )
  C_in <- if (!is.null(in_shape)) {
    tail(in_shape[!is.na(in_shape)], 1L)
  } else {
    1L
  }
  T_q <- as.integer(length(q_exprs) / C_in)
  T_kv <- as.integer(length(kv_exprs) / C_in)
  out_shape <- tryCatch(
    as.integer(unlist(l$output_shape)),
    error = function(e) NULL
  )
  C_out <- if (!is.null(out_shape)) {
    tail(out_shape[!is.na(out_shape)], 1L)
  } else {
    C_in
  }
  list(C_in = C_in, T_q = T_q, T_kv = T_kv, C_out = C_out)
}

# Build the output-projection bias accessor shared by MHA and GQA.
# Returns function(do_) -> character scalar ("0" when bias is absent).
.mha_output_bias_at <- function(wo) {
  function(do_) {
    if (!is.null(wo$bias) && length(wo$bias) >= do_) {
      format_numeric(wo$bias[do_])
    } else {
      "0"
    }
  }
}
