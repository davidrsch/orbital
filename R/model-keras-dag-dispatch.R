# Internal dispatch function for orbital_keras_dag_impl().
# Routes a single Keras layer to the appropriate .k3_* handler based on
# its class string. Called once per layer in the topological traversal loop.

.keras_dag_dispatch <- function(
  l,
  cls,
  cls_orig,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  # Shared argument list passed to every .k3_* handler.
  args <- list(
    l = l,
    lname = lname,
    topo_map = topo_map,
    expr_reg = expr_reg,
    state = state,
    weight_map = weight_map,
    output_layer_names = output_layer_names,
    last_dense = last_dense
  )

  if (grepl("input", cls)) {
    # Input layer: handled by the caller (no-op here; expression already
    # assigned in orbital_keras_dag_impl before dispatch is called).
    return(invisible(NULL))
  } else if (grepl("einsumdense", cls)) {
    do.call(.k3_einsumdense, args)
  } else if (grepl("dense", cls)) {
    do.call(.k3_dense, args)
  } else if (grepl("\\badd\\b", cls, perl = TRUE)) {
    do.call(.k3_add, args)
  } else if (grepl("\\bmultiply\\b", cls, perl = TRUE)) {
    do.call(.k3_multiply, args)
  } else if (
    grepl("\\baverage\\b", cls, perl = TRUE) && !grepl("pool|global", cls)
  ) {
    do.call(.k3_average, args)
  } else if (grepl("\\bmaximum\\b", cls, perl = TRUE)) {
    do.call(.k3_maximum, args)
  } else if (grepl("\\bminimum\\b", cls, perl = TRUE)) {
    do.call(.k3_minimum, args)
  } else if (grepl("\\bsubtract\\b", cls, perl = TRUE)) {
    do.call(.k3_subtract, args)
  } else if (grepl("\\bdot\\b", cls, perl = TRUE)) {
    do.call(.k3_dot, args)
  } else if (grepl("concatenate", cls)) {
    do.call(.k3_concatenate, args)
  } else if (grepl("batchnorm", cls)) {
    do.call(.k3_batchnorm, args)
  } else if (grepl("layernorm", cls)) {
    do.call(.k3_layernorm, args)
  } else if (grepl("prelu", cls)) {
    do.call(.k3_prelu, args)
  } else if (
    grepl(
      "dropout|flatten|reshape|gaussiannoise|activityregularization|\\bidentity\\b",
      cls,
      perl = TRUE
    )
  ) {
    do.call(.k3_dropout_passthru, args)
  } else if (grepl("\\bmasking\\b", cls, perl = TRUE)) {
    do.call(.k3_masking_passthru, args)
  } else if (grepl("leakyrelu", cls)) {
    do.call(.k3_leakyrelu, args)
  } else if (grepl("\\belu\\b", cls, perl = TRUE)) {
    do.call(.k3_elu, args)
  } else if (grepl("\\brelu\\b", cls, perl = TRUE)) {
    do.call(.k3_relu, args)
  } else if (
    grepl("\\bactivation\\b", cls, perl = TRUE) && !grepl("softmax", cls)
  ) {
    do.call(.k3_activation, args)
  } else if (grepl("globalaveragepool", cls)) {
    do.call(.k3_globalaveragepool, args)
  } else if (grepl("globalmaxpool", cls)) {
    do.call(.k3_globalmaxpool, args)
  } else if (grepl("adaptiveaveragepooling1d", cls)) {
    do.call(.k3_adaptiveaveragepooling1d, args)
  } else if (grepl("averagepooling1d", cls)) {
    do.call(.k3_averagepooling1d, args)
  } else if (grepl("adaptivemaxpooling1d", cls)) {
    do.call(.k3_adaptivemaxpooling1d, args)
  } else if (grepl("maxpooling1d", cls)) {
    do.call(.k3_maxpooling1d, args)
  } else if (grepl("upsampling1d", cls)) {
    do.call(.k3_upsampling1d, args)
  } else if (grepl("globalsumpooling", cls)) {
    do.call(.k3_globalsumpooling, args)
  } else if (grepl("instancenorm", cls)) {
    do.call(.k3_instancenorm, args)
  } else if (grepl("groupnorm", cls)) {
    do.call(.k3_groupnorm, args)
  } else if (grepl("rmsnormalization", cls)) {
    do.call(.k3_rmsnorm, args)
  } else if (grepl("\\bsoftmax\\b", cls, perl = TRUE)) {
    do.call(.k3_softmax, args)
  } else if (grepl("conv1dtranspose", cls)) {
    do.call(.k3_conv1dtranspose, args)
  } else if (grepl("depthwiseconv1d", cls)) {
    do.call(.k3_depthwiseconv1d, args)
  } else if (grepl("separableconv1d", cls)) {
    do.call(.k3_separableconv1d, args)
  } else if (grepl("convlstm1d", cls)) {
    do.call(.k3_convlstm1d, args)
  } else if (
    grepl("conv1d", cls) &&
      !grepl("depthwise|separable|transpose|2d|3d", cls)
  ) {
    do.call(.k3_conv1d, args)
  } else if (grepl("\\blstm\\b", cls, perl = TRUE)) {
    do.call(.k3_lstm, args)
  } else if (grepl("\\bgru\\b", cls, perl = TRUE)) {
    do.call(.k3_gru, args)
  } else if (grepl("bidirectional", cls)) {
    do.call(.k3_bidirectional, args)
  } else if (grepl("simplernn", cls)) {
    do.call(.k3_simplernn, args)
  } else if (grepl("unitnorm", cls)) {
    do.call(.k3_unitnorm, args)
  } else if (grepl("zeropadding1d", cls)) {
    do.call(.k3_zeropadding1d, args)
  } else if (grepl("permute", cls) && !grepl("2d|3d", cls)) {
    do.call(.k3_permute, args)
  } else if (grepl("cropping1d", cls)) {
    do.call(.k3_cropping1d, args)
  } else if (grepl("repeatvector", cls)) {
    do.call(.k3_repeatvector, args)
  } else if (grepl("additiveattention", cls)) {
    do.call(.k3_additiveattention, args)
  } else if (grepl("groupedqueryattention", cls)) {
    do.call(.k3_groupedqueryattention, args)
  } else if (
    grepl("\\battention\\b", cls, perl = TRUE) && !grepl("multihead", cls)
  ) {
    do.call(.k3_attention, args)
  } else if (grepl("timedistributed", cls)) {
    do.call(.k3_timedistributed, args)
  } else if (grepl("\\bembedding\\b", cls, perl = TRUE)) {
    do.call(.k3_embedding, args)
  } else if (grepl("multiheadattention", cls)) {
    do.call(.k3_multiheadattention, args)
  } else if (grepl("normalization", cls)) {
    do.call(.k3_normalization, args)
  } else if (grepl("categoryencoding", cls)) {
    do.call(.k3_categoryencoding, args)
  } else if (grepl("integerlookup", cls)) {
    do.call(.k3_integerlookup, args)
  } else if (grepl("rescaling", cls)) {
    do.call(.k3_rescaling, args)
  } else {
    # NOTE: The human-readable list below is intentionally hardcoded because it
    # appears verbatim in a user-facing error message. When you add a new
    # `grepl(...)` branch above, update this list so the message stays accurate.
    # See tests/testthat/test-model-keras-dag-softmax-axis-timedist.R for a
    # smoke-test that confirms the dispatcher rejects unknown classes.
    cli::cli_abort(c(
      "Unsupported layer type in Keras Functional model: {.cls {cls_orig}}.",
      "i" = paste(
        "orbital supports: Input, Dense, EinsumDense, Add, Multiply, Maximum, Minimum, Dot,",
        "Concatenate, BatchNormalization,",
        "LayerNormalization, InstanceNormalization, GroupNormalization,",
        "RMSNormalization, PReLU, LeakyReLU, ELU, ReLU, GlobalAveragePooling1D, GlobalMaxPooling1D,",
        "AdaptiveAveragePooling1D, AdaptiveMaxPooling1D,",
        "AveragePooling1D, MaxPooling1D, GlobalSumPooling1D,",
        "Conv1D, Conv1DTranspose, DepthwiseConv1D, SeparableConv1D,",
        "ConvLSTM1D, LSTM, GRU, Bidirectional(LSTM/GRU), SimpleRNN,",
        "UnitNormalization, ZeroPadding1D, Cropping1D, RepeatVector, Permute,",
        "Embedding, TimeDistributed(Dense),",
        "GroupedQueryAttention, MultiHeadAttention, Attention, AdditiveAttention, Subtract,",
        "UpSampling1D, Dropout, SpatialDropout1D, GaussianDropout, GaussianNoise,",
        "AlphaDropout, Masking, Flatten, Reshape, Activation, Softmax,",
        "Normalization, CategoryEncoding, IntegerLookup, Rescaling."
      ),
      "i" = "Please file an issue: {.url https://github.com/davidrsch/orbital/issues/14}"
    ))
  }
}
