# Shared RNN unroll dispatcher for LSTM and GRU.
# Per-type kernels live in:
#   model-keras-recurrent-unroll-lstm.R  (.unroll_lstm)
#   model-keras-recurrent-unroll-gru.R   (.unroll_gru)
# Gate variable names: a_input, a_forget, a_cell, a_output (LSTM); z, r, ht (GRU).
# Called by .k3_lstm(), .k3_gru(), .k3_simplernn() in model-keras-recurrent.R.

.unroll_rnn <- function(ul, ul_in_exprs, ul_pfx) {
  inner_cls <- tolower(class(ul)[1L])
  is_lstm_inner <- grepl("\\blstm\\b", inner_cls, perl = TRUE)
  ul_wts <- ul$get_weights()
  ul_cfg <- tryCatch(ul$get_config(), error = function(e) list())
  ul_return_seq <- isTRUE(
    tryCatch(as.logical(ul_cfg$return_sequences), error = function(e) FALSE)
  )
  ul_gate_act <- tryCatch(
    tolower(as.character(ul_cfg$recurrent_activation %||% "sigmoid")),
    error = function(e) "sigmoid"
  )
  ul_cell_act <- tryCatch(
    tolower(as.character(ul_cfg$activation %||% "tanh")),
    error = function(e) "tanh"
  )

  if (is_lstm_inner) {
    .unroll_lstm(
      ul_in_exprs = ul_in_exprs,
      ul_wts = ul_wts,
      ul_pfx = ul_pfx,
      ul_gate_act = ul_gate_act,
      ul_cell_act = ul_cell_act,
      ul_return_seq = ul_return_seq
    )
  } else {
    .unroll_gru(
      ul_in_exprs = ul_in_exprs,
      ul_wts = ul_wts,
      ul_cfg = ul_cfg,
      ul_pfx = ul_pfx,
      ul_gate_act = ul_gate_act,
      ul_cell_act = ul_cell_act,
      ul_return_seq = ul_return_seq
    )
  }
}

# Flatten a 2-row bias matrix (input bias + recurrent bias) into a single
# bias vector by summing the two rows.  Handles the case where Keras stores
# bias as either a (2, units) matrix or a plain flat vector.
.flatten_rnn_bias <- function(raw_b) {
  if (!is.null(dim(raw_b)) && length(dim(raw_b)) == 2L) {
    as.numeric(raw_b[1L, ]) + as.numeric(raw_b[2L, ])
  } else {
    as.numeric(raw_b)
  }
}

# Shared masking-detection helper used by LSTM and GRU handlers.
# Walk topo_map backwards from lname to detect whether any upstream layer
# is a Keras Masking layer or an Embedding with mask_zero = TRUE.
# Returns TRUE if masking is active, FALSE otherwise.
.detect_masking_upstream <- function(lname, topo_map) {
  visited <- character(0L)
  to_visit <- topo_map[[lname]] %||% character(0L)
  while (length(to_visit) > 0L) {
    nm <- to_visit[[1L]]
    to_visit <- to_visit[-1L]
    if (nm %in% visited) {
      next
    }
    visited <- c(visited, nm)
    if (grepl("masking", nm, ignore.case = TRUE)) {
      return(TRUE)
    }
    to_visit <- c(to_visit, topo_map[[nm]] %||% character(0L))
  }
  FALSE
}

# Shared guard for stateful RNN and upstream masking — used by LSTM, GRU, and SimpleRNN.
# unit_type: human-readable name ("LSTM" | "GRU" | "SimpleRNN") for error messages.
.check_stateful_and_masking <- function(cfg_l, lname, topo_map, unit_type) {
  if (isTRUE(tryCatch(as.logical(cfg_l$stateful), error = function(e) FALSE))) {
    cli::cli_abort(c(
      "{unit_type} layer {.val {lname}}: stateful = TRUE is not supported by orbital.",
      "i" = "Only stateless {unit_type}s (stateful = FALSE, the Keras default) can be unrolled into SQL."
    ))
  }
  if (.detect_masking_upstream(lname, topo_map)) {
    cli::cli_abort(
      c(
        "{unit_type} layer {.val {lname}}: a masking layer was detected upstream.",
        "i" = "Sequence masks cannot be applied in the generated SQL.",
        "i" = "Predictions for variable-length (padded) sequences would differ from Keras.",
        "i" = "Remove the upstream Masking / Embedding(mask_zero=TRUE) or pre-mask inputs before calling orbital()."
      ),
      .class = "orbital_masking_ignored"
    )
  }
  invisible(NULL)
}


# Shared guard: reject any attention layer whose sequence path includes a
# Keras Masking layer or an Embedding with mask_zero = TRUE.  Attention masks
# (including the `attention_mask` kwarg and upstream mask propagation) are not
# applied in the generated SQL, which would silently produce wrong results
# on padded sequences.
.check_attention_mask <- function(lname, topo_map, layer_kind) {
  if (.detect_masking_upstream(lname, topo_map)) {
    cli::cli_abort(c(
      "{layer_kind} layer {.val {lname}}: attention masking is not supported by orbital.",
      "i" = "A Keras Masking layer or Embedding(mask_zero = TRUE) was detected upstream.",
      "i" = "Attention masks cannot be applied in the generated SQL; padded sequences would produce silently wrong results.",
      "i" = "Remove upstream masking, or pre-mask inputs before calling orbital()."
    ))
  }
  invisible(NULL)
}

# Shared setup helper for gated RNN layers (LSTM, GRU).
# Extracts inputs, weights, and activation config; calls .check_stateful_and_masking.
# Returns a list with: in_exprs, wts, kernel, rkernel, I_feat, T_len, cfg_l,
#   gate_act, cell_act, return_seq, go_backwards.
.rnn_extract_gated_setup <- function(l, lname, topo_map, expr_reg, layer_abbr) {
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  wts <- l$get_weights()
  kernel <- wts[[1L]]
  rkernel <- wts[[2L]]
  I_feat <- nrow(kernel)
  T_len <- as.integer(length(in_exprs) / I_feat)
  cfg_l <- .k3_safe_get_config(l, lname)
  .check_stateful_and_masking(cfg_l, lname, topo_map, layer_abbr)
  gate_act <- tryCatch(
    tolower(as.character(cfg_l$recurrent_activation %||% "sigmoid")),
    error = function(e) "sigmoid"
  )
  cell_act <- tryCatch(
    tolower(as.character(cfg_l$activation %||% "tanh")),
    error = function(e) "tanh"
  )
  return_seq <- isTRUE(
    tryCatch(as.logical(cfg_l$return_sequences), error = function(e) FALSE)
  )
  go_backwards <- isTRUE(cfg_l$go_backwards)
  list(
    in_exprs = in_exprs,
    wts = wts,
    kernel = kernel,
    rkernel = rkernel,
    I_feat = I_feat,
    T_len = T_len,
    cfg_l = cfg_l,
    gate_act = gate_act,
    cell_act = cell_act,
    return_seq = return_seq,
    go_backwards = go_backwards
  )
}
