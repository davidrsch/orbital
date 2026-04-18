# Shared helper functions for Keras MultiHeadAttention / GroupedQueryAttention.
# These helpers centralize 3-D kernel layout resolution and 2-D bias accessors
# used by `model-keras-mha-standard.R` and `model-keras-mha-gqa.R`.

#' Resolve a 3-D MHA / GQA projection-kernel layout.
#'
#' Accepts kernels of shape `(C_in, d2, d3)` or its transpose
#' `(C_in, d3, d2)`, aborts on ambiguous (`d2 == d3`) or unrecognised layouts,
#' and returns an indexer `function(c_i, i2, i3)` into the canonical
#' `(C_in, d2, d3)` layout.
#'
#' @param k The kernel array.
#' @param d2,d3 The two non-`C_in` dimensions to disambiguate against.
#' @param C_in Channel-in dimension.
#' @param lname Keras layer name (for error context).
#' @param name Short label for the kernel (e.g. `"Q"`, `"K"`).
#' @param layer_abbr Abbreviation for the calling layer family (`"MHA"` / `"GQA"`).
#' @param ambiguity_hint Optional message appended when `d2 == d3`.
#' @return A scalar indexer `function(c_i, i2, i3)`.
#' @keywords internal
#' @noRd
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

#' 2-D bias accessor for MHA / GQA projection sub-layers.
#'
#' Accepts a `(d2, d3)` or transposed `(d3, d2)` bias matrix; returns `"0"`
#' when the bias is `NULL` or has an unrecognised shape.
#'
#' @param b Bias array (`NULL` if absent).
#' @param d2,d3 The two expected dimensions.
#' @return `function(i2, i3) -> character` (formatted numeric or `"0"`).
#' @keywords internal
#' @noRd
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

#' Extract sequence-shape variables shared by MHA and GQA.
#'
#' Reads `l$input_shape[[1L]]` and `l$output_shape` to derive the
#' channel-in / time / channel-out triplet.
#'
#' @param l The live attention layer.
#' @param q_exprs,kv_exprs Inbound expression vectors for the Q and K/V branches.
#' @return A named list with elements `C_in`, `T_q`, `T_kv`, `C_out`.
#' @keywords internal
#' @noRd
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

#' Build the output-projection bias accessor shared by MHA and GQA.
#'
#' @param wo The output-projection weight bundle (with `$bias`).
#' @return `function(do_) -> character` (formatted numeric or `"0"`).
#' @keywords internal
#' @noRd
.mha_output_bias_at <- function(wo) {
  function(do_) {
    if (!is.null(wo$bias) && length(wo$bias) >= do_) {
      format_numeric(wo$bias[do_])
    } else {
      "0"
    }
  }
}

#' Resolve the output-projection kernel layout.
#'
#' Accepts either `(num_heads, d_v, C_out)` or `(C_out, num_heads, d_v)`;
#' aborts otherwise.
#'
#' @param wo The output-projection weight bundle.
#' @param num_heads,d_v,C_out Expected dimensions.
#' @param lname Keras layer name (for error context).
#' @param layer_abbr Abbreviation for the calling layer family.
#' @return A list with elements `wo_at = function(h, dv, do_)` and `C_out_w`.
#' @keywords internal
#' @noRd
.mha_resolve_output_kernel <- function(
  wo,
  num_heads,
  d_v,
  C_out,
  lname,
  layer_abbr = "MHA"
) {
  wo_d <- dim(wo$kernel)
  if (length(wo_d) == 3L && wo_d[1L] == num_heads && wo_d[2L] == d_v) {
    return(list(
      wo_at = function(h, dv, do_) wo$kernel[h, dv, do_],
      C_out_w = wo_d[3L]
    ))
  }
  if (length(wo_d) == 3L && wo_d[1L] == C_out && wo_d[2L] == num_heads) {
    return(list(
      wo_at = function(h, dv, do_) wo$kernel[do_, h, dv],
      C_out_w = wo_d[1L]
    ))
  }
  cli::cli_abort(
    "{layer_abbr} {.val {lname}}: output kernel dims ({paste(wo_d, collapse = 'x')}) unrecognised."
  )
}

#' Build a `(t, c)` -> backticked column-name accessor.
#'
#' Used to address the flat time-step major (`T x C_in`) expression vectors
#' produced by the inbound layers of MHA / GQA.
#'
#' @param exprs Character vector of inbound expression-column names.
#' @param C_in Channel-in dimension (used as the time-step stride).
#' @return `function(t, c) -> backticked column name`.
#' @keywords internal
#' @noRd
.mha_input_accessor <- function(exprs, C_in) {
  force(exprs)
  force(C_in)
  function(t, c) backtick(exprs[(t - 1L) * C_in + c])
}
