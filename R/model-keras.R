# orbital methods for all keras-based models (legacy keras R package and keras3/kerasnip)

# Internal: parse inbound layer names from one layer's config entry.
# Handles keras3 new format ({args: [{keras_history: ["name", 0, 0]}]})
# and old keras format ([[["name", 0, 0], ...]]).
.keras_parse_inbound <- function(layer_cfg) {
  nodes <- layer_cfg[["inbound_nodes"]]
  if (is.null(nodes) || length(nodes) == 0L) {
    return(character(0L))
  }

  first_node <- nodes[[1L]]

  # keras3 format: first_node = list(args = list(...), kwargs = list())
  args <- first_node[["args"]]
  if (!is.null(args) && is.list(args)) {
    res <- vapply(
      args,
      function(arg) {
        hist <- arg[["keras_history"]]
        if (is.null(hist)) {
          return(NA_character_)
        }
        lyr <- hist[[1L]]
        if (is.character(lyr)) {
          lyr
        } else {
          tryCatch(lyr$name, error = function(e) {
            as.character(lyr)[1L]
          })
        }
      },
      character(1L)
    )
    return(res[!is.na(res)])
  }

  # Old keras format: first_node is a list of [layer_name, node_idx, tensor_idx] triples
  if (
    is.list(first_node) &&
      length(first_node) > 0L &&
      is.list(first_node[[1L]])
  ) {
    return(vapply(
      first_node,
      function(spec) {
        n <- spec[[1L]]
        if (is.character(n)) n else as.character(n)[1L]
      },
      character(1L)
    ))
  }

  character(0L)
}


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
    if (grepl("dense", tolower(class(l)[1L]))) {
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
    } else if (grepl("einsumdense", cls)) {
      .k3_einsumdense(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("dense", cls)) {
      .k3_dense(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("\\badd\\b", cls, perl = TRUE)) {
      .k3_add(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("\\bmultiply\\b", cls, perl = TRUE)) {
      .k3_multiply(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (
      grepl("\\baverage\\b", cls, perl = TRUE) && !grepl("pool|global", cls)
    ) {
      .k3_average(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("\\bmaximum\\b", cls, perl = TRUE)) {
      .k3_maximum(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("\\bminimum\\b", cls, perl = TRUE)) {
      .k3_minimum(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("\\bsubtract\\b", cls, perl = TRUE)) {
      .k3_subtract(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("\\bdot\\b", cls, perl = TRUE)) {
      .k3_dot(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("concatenate", cls)) {
      .k3_concatenate(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("batchnorm", cls)) {
      .k3_batchnorm(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("layernorm", cls)) {
      .k3_layernorm(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("prelu", cls)) {
      .k3_prelu(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("dropout|flatten|reshape", cls)) {
      .k3_dropout_passthru(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("leakyrelu", cls)) {
      .k3_leakyrelu(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("\\belu\\b", cls, perl = TRUE)) {
      .k3_elu(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("\\brelu\\b", cls, perl = TRUE)) {
      .k3_relu(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (
      grepl("\\bactivation\\b", cls, perl = TRUE) && !grepl("softmax", cls)
    ) {
      .k3_activation(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("globalaveragepool", cls)) {
      .k3_globalaveragepool(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("globalmaxpool", cls)) {
      .k3_globalmaxpool(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("adaptiveaveragepooling1d", cls)) {
      .k3_adaptiveaveragepooling1d(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("averagepooling1d", cls)) {
      .k3_averagepooling1d(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("adaptivemaxpooling1d", cls)) {
      .k3_adaptivemaxpooling1d(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("maxpooling1d", cls)) {
      .k3_maxpooling1d(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("upsampling1d", cls)) {
      .k3_upsampling1d(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("globalsumpooling", cls)) {
      .k3_globalsumpooling(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("instancenorm", cls)) {
      .k3_instancenorm(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("groupnorm", cls)) {
      .k3_groupnorm(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("rmsnormalization", cls)) {
      .k3_rmsnorm(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("\\bsoftmax\\b", cls, perl = TRUE)) {
      .k3_softmax(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("conv1dtranspose", cls)) {
      .k3_conv1dtranspose(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("depthwiseconv1d", cls)) {
      .k3_depthwiseconv1d(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("separableconv1d", cls)) {
      .k3_separableconv1d(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("convlstm1d", cls)) {
      .k3_convlstm1d(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (
      grepl("conv1d", cls) && !grepl("depthwise|separable|transpose|2d|3d", cls)
    ) {
      .k3_conv1d(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("\\blstm\\b", cls, perl = TRUE)) {
      .k3_lstm(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("\\bgru\\b", cls, perl = TRUE)) {
      .k3_gru(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("bidirectional", cls)) {
      .k3_bidirectional(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("simplernn", cls)) {
      .k3_simplernn(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("unitnorm", cls)) {
      .k3_unitnorm(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("zeropadding1d", cls)) {
      .k3_zeropadding1d(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("permute", cls) && !grepl("2d|3d", cls)) {
      .k3_permute(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("cropping1d", cls)) {
      .k3_cropping1d(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("repeatvector", cls)) {
      .k3_repeatvector(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("additiveattention", cls)) {
      .k3_additiveattention(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("groupedqueryattention", cls)) {
      .k3_groupedqueryattention(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (
      grepl("\\battention\\b", cls, perl = TRUE) && !grepl("multihead", cls)
    ) {
      .k3_attention(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("timedistributed", cls)) {
      .k3_timedistributed(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("\\bembedding\\b", cls, perl = TRUE)) {
      .k3_embedding(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else if (grepl("multiheadattention", cls)) {
      .k3_multiheadattention(
        l,
        lname,
        topo_map,
        expr_reg,
        state,
        weight_map,
        output_layer_names,
        last_dense
      )
    } else {
      cli::cli_abort(c(
        "Unsupported layer type in Keras Functional model: {.cls {cls_orig}}.",
        "i" = paste(
          "orbital supports: Dense, EinsumDense, Add, Concatenate, BatchNormalization,",
          "LayerNormalization, InstanceNormalization, GroupNormalization,",
          "RMSNormalization, PReLU, GlobalAveragePooling1D, GlobalMaxPooling1D,",
          "AdaptiveAveragePooling1D, AdaptiveMaxPooling1D,",
          "AveragePooling1D, MaxPooling1D, GlobalSumPooling1D,",
          "Conv1D, ConvLSTM1D, LSTM, GRU, Bidirectional(LSTM/GRU), SimpleRNN,",
          "UnitNormalization, ZeroPadding1D, Embedding, TimeDistributed(Dense),",
          "GroupedQueryAttention, MultiHeadAttention, Attention, AdditiveAttention, Subtract,",
          "UpSampling1D, Dropout, SpatialDropout1D, GaussianDropout,",
          "AlphaDropout, Flatten, Reshape, Activation, Softmax."
        ),
        "i" = "Please file an issue: {.url https://github.com/davidrsch/orbital/issues/14}"
      ))
    }
  }

  all_exprs <- state$all_exprs
  out_pre_act_map <- state$out_pre_act_map
  out_pre_act <- state$out_pre_act

  if (length(out_pre_act_map) == 0L) {
    cli::cli_abort(
      "Could not identify any output Dense layer in the Keras model."
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


orbital_keras_impl <- function(
  x,
  mode,
  type,
  lvl,
  prefix,
  feature_names = NULL
) {
  all_weights <- x$get_weights()
  all_layers <- x$layers

  # Detect layers that require DAG traversal (merge, normalisation, learned activations).
  nl_re <- paste0(
    c(
      # Merge layers
      "\\badd\\b",
      "concatenate",
      "\\bmultiply\\b",
      "\\baverage\\b",
      "\\bmaximum\\b",
      "\\bminimum\\b",
      "\\bdot\\b",
      "\\bsubtract\\b",
      # Normalisation
      "batchnorm",
      "layernorm",
      "instancenorm",
      "groupnorm",
      "rmsnormalization",
      "unitnorm",
      # Learned activations
      "prelu",
      "leakyrelu",
      "\\belu\\b",
      "\\brelu\\b",
      "\\bactivation\\b",
      "\\bsoftmax\\b",
      # Pooling
      "globalaveragepool",
      "globalmaxpool",
      "adaptiveaveragepooling1d",
      "adaptivemaxpooling1d",
      "averagepooling1d",
      "maxpooling1d",
      "globalsumpooling",
      # Recurrent
      "\\blstm\\b",
      "\\bgru\\b",
      "convlstm1d",
      "bidirectional",
      "simplernn",
      # Convolution
      "conv1d",
      "conv1dtranspose",
      "depthwiseconv1d",
      "separableconv1d",
      # Attention (specific patterns before the generic guard)
      "additiveattention",
      "groupedqueryattention",
      "\\battention\\b",
      "multiheadattention",
      # Spatial / sequence
      "zeropadding1d",
      "permute",
      "cropping1d",
      "repeatvector",
      "\\bembedding\\b",
      "timedistributed",
      "upsampling1d"
    ),
    collapse = "|"
  )
  nl_excl_re <- paste0(
    c(
      "\\bdense\\b",
      "input",
      "flatten",
      "reshape",
      "dropout",
      "depthwiseconv2d",
      "separableconv2d",
      "2d",
      "3d"
    ),
    collapse = "|"
  )
  non_linear_layers <- Filter(
    function(l) {
      cls <- tolower(class(l)[1L])
      grepl(nl_re, cls, perl = TRUE) && !grepl(nl_excl_re, cls, perl = TRUE)
    },
    all_layers
  )

  if (length(non_linear_layers) > 0L) {
    return(
      orbital_keras_dag_impl(
        x,
        mode,
        type,
        lvl,
        prefix,
        feature_names,
        all_weights,
        all_layers
      )
    )
  }

  # Linear model: optimised sequential traversal (no topology introspection needed)
  n_dense <- length(all_weights) / 2L # each Dense layer has kernel + bias

  dense_layers <- Filter(
    function(l) grepl("dense", tolower(class(l)[1L])),
    all_layers
  )

  n_in <- ncol(all_weights[[1L]])
  input_names <- if (!is.null(feature_names)) {
    feature_names
  } else {
    paste0("orbital_feature_", seq_len(n_in))
  }

  all_exprs <- list()
  current_names <- input_names

  for (i in seq_len(n_dense)) {
    kernel <- t(all_weights[[2L * i - 1L]]) # (n_out_i x n_in_i)
    bias <- as.numeric(all_weights[[2L * i]])

    activation_config <- tryCatch(
      dense_layers[[i]]$get_config()$activation,
      error = function(e) NULL
    )
    activation <- if (is.null(activation_config)) {
      "linear"
    } else if (is.list(activation_config)) {
      tolower(as.character(activation_config$class_name)[1L])
    } else {
      tolower(as.character(activation_config)[1L])
    }
    if (!nzchar(activation)) {
      activation <- "linear"
    }
    act_alpha <- if (
      is.list(activation_config) &&
        !is.null(activation_config[["config"]])
    ) {
      activation_config[["config"]][["alpha"]]
    } else {
      NULL
    }

    pre_act <- build_mlp_pre_act(kernel, bias, current_names)

    if (i < n_dense) {
      act_exprs <- vapply(
        pre_act,
        function(z) activation_expr(activation, z, alpha = act_alpha),
        character(1)
      )
      layer_names <- paste0(
        "orbital_mlp_l",
        i,
        "_h",
        seq_len(nrow(kernel))
      )
      all_exprs[[i]] <- stats::setNames(act_exprs, layer_names)
      current_names <- layer_names
    } else {
      out_pre_act <- pre_act
    }
  }

  hidden_exprs <- unlist(all_exprs, use.names = TRUE)
  n_out <- length(out_pre_act)

  if (mode == "regression") {
    c(hidden_exprs, stats::setNames(out_pre_act[1L], prefix))
  } else if (n_out == 1L) {
    sigmoid_expr <- activation_expr("sigmoid", out_pre_act[1L])
    c(hidden_exprs, binary_from_prob(sigmoid_expr, type, lvl))
  } else {
    logit_exprs <- stats::setNames(out_pre_act, lvl)
    c(hidden_exprs, multiclass_from_logits(logit_exprs, type, lvl))
  }
}


#' @method orbital keras.engine.sequential.Sequential
#' @export
orbital.keras.engine.sequential.Sequential <- function(
  x,
  ...,
  mode = c("regression", "classification"),
  type = NULL,
  lvl = NULL,
  prefix = ".pred"
) {
  mode <- rlang::arg_match(mode)
  type <- default_type(type)
  extra <- list(...)
  orbital_keras_impl(
    x,
    mode = mode,
    type = type,
    lvl = lvl,
    prefix = prefix,
    feature_names = extra$feature_names
  )
}

#' @method orbital keras.src.models.sequential.Sequential
#' @export
orbital.keras.src.models.sequential.Sequential <- function(
  x,
  ...,
  mode = c("regression", "classification"),
  type = NULL,
  lvl = NULL,
  prefix = ".pred"
) {
  mode <- rlang::arg_match(mode)
  type <- default_type(type)
  extra <- list(...)
  orbital_keras_impl(
    x,
    mode = mode,
    type = type,
    lvl = lvl,
    prefix = prefix,
    feature_names = extra$feature_names
  )
}

#' @method orbital keras.src.models.functional.Functional
#' @export
orbital.keras.src.models.functional.Functional <- function(
  x,
  ...,
  mode = c("regression", "classification"),
  type = NULL,
  lvl = NULL,
  prefix = ".pred"
) {
  mode <- rlang::arg_match(mode)
  type <- default_type(type)
  extra <- list(...)
  orbital_keras_impl(
    x,
    mode = mode,
    type = type,
    lvl = lvl,
    prefix = prefix,
    feature_names = extra$feature_names
  )
}
