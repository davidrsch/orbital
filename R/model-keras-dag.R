# Internal DAG/Functional API implementation for orbital keras methods.
# Provides orbital_keras_dag_impl() called by orbital_keras_impl() in
# model-keras-impl.R.
#
# Helper functions split into separate files:
#   .keras_parse_inbound(), .k3_masking_passthru()  -> model-keras-dag-helpers.R
#   .keras_dag_dispatch()                            -> model-keras-dag-dispatch.R

# Internal: topological/DAG traversal for Functional API models with merge layers.

orbital_keras_dag_impl <- function(
  x,
  mode,
  type,
  lvl,
  prefix,
  feature_names,
  all_weights,
  all_layers
) {
  # Build Dense weight map: layer_name → list(kernel, bias)
  # Use per-layer get_weights() so BatchNorm/PReLU weights don't skew indices.
  weight_map <- list()
  for (l in all_layers) {
    cls_l <- tolower(class(l)[1L])
    if (grepl("dense", cls_l) && !grepl("einsumdense", cls_l)) {
      wts <- l$get_weights()
      if (length(wts) >= 2L) {
        weight_map[[l$name]] <- list(
          kernel = t(wts[[1L]]),
          bias = as.numeric(wts[[2L]])
        )
      }
    }
  }
  last_dense <- if (length(weight_map) > 0L) {
    tail(names(weight_map), 1L)
  } else {
    ""
  }

  n_in <- ncol(weight_map[[names(weight_map)[1L]]]$kernel)
  input_names <- if (!is.null(feature_names)) {
    feature_names
  } else {
    paste0("orbital_feature_", seq_len(n_in))
  }

  # Parse layer topology from model config
  cfg <- tryCatch(x$get_config(), error = function(e) NULL)
  layers_cfg <- if (!is.null(cfg)) {
    if (!is.null(cfg$config) && !is.null(cfg$config$layers)) {
      cfg$config$layers
    } else if (!is.null(cfg$layers)) {
      cfg$layers
    } else {
      list()
    }
  } else {
    list()
  }

  # Determine output layer names from model config (Functional API multi-output)
  raw_out <- if (!is.null(cfg)) {
    if (!is.null(cfg$config$output_layers)) {
      cfg$config$output_layers
    } else if (!is.null(cfg$output_layers)) {
      cfg$output_layers
    } else {
      NULL
    }
  } else {
    NULL
  }
  output_layer_names <- if (!is.null(raw_out) && length(raw_out) > 0L) {
    unique(vapply(
      raw_out,
      function(spec) {
        n <- if (is.list(spec)) spec[[1L]] else spec
        as.character(n)[1L]
      },
      character(1L)
    ))
  } else {
    cli::cli_warn(
      c(
        paste0(
          "Could not determine output layer names from model config; ",
          "falling back to last Dense layer ({.val {last_dense}})."
        ),
        "i" = "For multi-output models, verify that predictions are correct."
      ),
      .class = "orbital_output_fallback"
    )
    last_dense # fallback: treat last Dense as the single output
  }

  topo_map <- vector("list", length(layers_cfg))
  for (k in seq_along(layers_cfg)) {
    lc <- layers_cfg[[k]]
    lname <- tryCatch(
      {
        n <- if (!is.null(lc$config) && !is.null(lc$config$name)) {
          lc$config$name
        } else {
          lc$name
        }
        as.character(n)[1L]
      },
      error = function(e) ""
    )
    if (nzchar(lname)) {
      topo_map[[lname]] <- .keras_parse_inbound(lc)
    }
  }

  # Expression register: layer_name → character vector of expression names
  expr_reg <- new.env(hash = TRUE, parent = emptyenv())

  # State environment: mutable shared state passed to all handlers by reference.
  state <- new.env(parent = emptyenv())
  state$all_exprs <- list()
  state$out_pre_act_map <- list()
  state$out_pre_act <- NULL

  for (l in all_layers) {
    cls_orig <- class(l)[1L]
    cls <- tolower(cls_orig)
    lname <- l$name

    if (grepl("input", cls)) {
      assign(lname, input_names, envir = expr_reg)
    } else {
      .keras_dag_dispatch(
        l, cls, cls_orig, lname,
        topo_map, expr_reg, state,
        weight_map, output_layer_names, last_dense
      )
    }
  }

  all_exprs <- state$all_exprs
  out_pre_act_map <- state$out_pre_act_map
  out_pre_act <- state$out_pre_act

  if (length(out_pre_act_map) == 0L) {
    out_layer_types <- vapply(
      all_layers,
      function(l) {
        if (l$name %in% output_layer_names) class(l)[1L] else NA_character_
      },
      character(1L)
    )
    out_layer_types <- out_layer_types[!is.na(out_layer_types)]
    cli::cli_abort(
      c(
        "Could not identify any output Dense layer in the Keras model.",
        "i" = if (length(out_layer_types) > 0L) {
          paste0(
            "The output layer(s) identified are of type: ",
            paste(out_layer_types, collapse = ", "),
            "."
          )
        } else {
          "No output layer names could be resolved from the model config."
        },
        "i" = paste0(
          "orbital requires the final output layer to be a Dense (or EinsumDense) layer.",
          " Standalone Activation, Softmax, or BatchNormalization output layers",
          " are not supported as the last layer."
        )
      )
    )
  }

  hidden_exprs <- unlist(all_exprs, use.names = TRUE)

  # Combine pre-activations from all output layers in config order
  all_out_pre_act <- unlist(
    out_pre_act_map[output_layer_names],
    use.names = FALSE
  )
  n_out <- length(all_out_pre_act)

  if (mode == "regression") {
    if (n_out == 1L) {
      c(hidden_exprs, stats::setNames(all_out_pre_act[1L], prefix))
    } else {
      # Multiple regression outputs: name them prefix_1, prefix_2, ...
      out_nms <- if (length(all_out_pre_act) == length(lvl) && !is.null(lvl)) {
        lvl
      } else {
        paste0(prefix, "_", seq_along(all_out_pre_act))
      }
      c(hidden_exprs, stats::setNames(all_out_pre_act, out_nms))
    }
  } else if (n_out == 1L) {
    c(
      hidden_exprs,
      binary_from_prob(
        activation_expr("sigmoid", all_out_pre_act[1L]),
        type,
        lvl
      )
    )
  } else {
    c(
      hidden_exprs,
      multiclass_from_logits(
        stats::setNames(all_out_pre_act, lvl),
        type,
        lvl
      )
    )
  }
}
