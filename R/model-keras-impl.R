# Sequential routing and linear-path implementation for orbital keras methods.
# Provides orbital_keras_impl() which dispatches to orbital_keras_dag_impl()
# (in model-keras-dag.R) for non-linear functional models.

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
      "\\bbatchnormalization\\b",
      "\\blayernormalization\\b",
      "\\binstancenormalization\\b",
      "\\bgroupnormalization\\b",
      "\\brmsnormalization\\b",
      "\\bunitnormalization\\b",
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
      "einsumdense",
      "\\blstm\\b",
      "\\bgru\\b",
      "convlstm1d",
      "bidirectional",
      "simplernn",
      # Convolution
      "\\bconv1d\\b",
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
      "upsampling1d",
      "\\bmasking\\b",
      # Preprocessing layers (Normalization, IntegerLookup, CategoryEncoding, etc.)
      "preprocessing"
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
      "3d",
      "gaussiannoise",
      "activityregularization",
      "\\bidentity\\b"
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
  dense_layers <- Filter(
    function(l) grepl("dense", tolower(class(l)[1L])),
    all_layers
  )
  n_dense <- length(dense_layers)

  n_in <- ncol(all_weights[[1L]])
  input_names <- if (!is.null(feature_names)) {
    feature_names
  } else {
    paste0("orbital_feature_", seq_len(n_in))
  }

  all_exprs <- list()
  current_names <- input_names

  for (i in seq_len(n_dense)) {
    wts_i <- dense_layers[[i]]$get_weights()
    kernel <- t(wts_i[[1L]]) # (n_out_i x n_in_i)
    bias <- if (length(wts_i) >= 2L) {
      as.numeric(wts_i[[2L]])
    } else {
      numeric(nrow(kernel))
    }

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
    act_default_value <- if (
      is.list(activation_config) &&
        !is.null(activation_config[["config"]])
    ) {
      activation_config[["config"]][["default_value"]] %||% 0
    } else {
      0
    }

    pre_act <- build_mlp_pre_act(kernel, bias, current_names)

    if (i < n_dense) {
      act_exprs <- vapply(
        pre_act,
        function(z) {
          activation_expr(
            activation,
            z,
            alpha = act_alpha,
            default_value = act_default_value
          )
        },
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
