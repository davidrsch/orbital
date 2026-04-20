# orbital (development version)

## New features

- **brulee:** `mlp()` fits with `hidden_units = c(...)` of arbitrary length are now translated (previously aborted at 2 hidden layers). Per-layer activations are honoured via the `activation` argument as a vector or scalar (recycled). (#B-01)
- **Keras 3:** `Attention(use_scale = TRUE)` and `MultiHeadAttention(use_causal_mask = TRUE)` are now supported. (#R-03)
- **brulee / H2O:** additional activations supported — `sparse_sigmoid` and `sparse_plus` (column-wise) for brulee/Keras; `ExpRectifier` / `ExpRectifierWithDropout` for H2O. (#R-G5, #H-G1)
- **Keras 3:** improved error messages for the still-unsupported `sparsemax` and `glu` activations.

## Bug fixes

- **H2O:** abort when `standardize = TRUE` but normalization vectors are missing from the fit object, instead of silently producing wrong predictions. (#H-01)
- **nnet:** validate the weight-vector length before reading it; emit an informative error when passed a `bag_mlp` object pointing users at `parsnip::extract_fit_engine()`. (#N-01, #N-02)
- **H2O:** warn when `@parameters$activation` is `NULL` so the silent `Rectifier` default cannot mask a corrupted model. (#H-02)
- **Keras 3:** explicit guard against `EinsumDense` layers misrouting to the linear `Sequential` dispatch path. (#R-02)
- **H2O:** narrow the weight-discovery `tryCatch` so genuine connection / auth / version errors are no longer silently swallowed as end-of-matrix signals. (#H-03)

## Documentation

- Corrected the `keras3-models.Rmd` supported-layers table to reflect current support for `LSTM`, `GRU`, `Conv1D`, `Attention`, `AdditiveAttention`, and `MultiHeadAttention`. (#R-01)
- Documented nnet backend scope (single hidden layer, `skip`, `linout`, `softmax`, `censored`, `bag_mlp`) and Keras `Embedding` / `Reshape` behaviour in the supported-models vignette. (#N-04, #N-05, #R-05, #R-06)
- Documented H2O `standardize = TRUE` best practice and multiclass `lvl` ordering in the `orbital.H2ODeepLearningModel` roxygen section. (#H-05)
- Documented the full brulee activation vocabulary and deep-MLP support in the `orbital.brulee_mlp` roxygen section. (#B-04)

- `estimate_orbital_size()` is a new function that quickly estimates the character count of the orbital expression for a model without generating it. (#144)

- `step_dummy()` and `step_indicate_na()` now generate SQL compatible with Snowflake and other databases that don't support casting booleans directly to numeric types. (#145)

- `orbital()` now supports Keras3 neural networks (Sequential and Functional/DAG models), including dense layers, normalization (Batch/Layer/Instance/Group), pooling (GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D, MaxPooling1D, GlobalSumPooling1D), merge layers (Add, Concatenate), PReLU, and all standard activation functions. `GlobalSumPooling1D` is a Keras3-native layer with no direct ONNX counterpart; it is not available when translating PyTorch/ONNX models.

- `orbital()` now supports `qrnn` models via [qrnn::qrnn.fit()]. Because `qrnn.fit()` returns a plain list with no S3 class, wrap the result with `as_qrnn_fit()` before passing to `orbital()`.

- `orbital()` now supports `mlp(engine = "brulee")` and `brulee::brulee_mlp_two_layer()` models (up to 2 hidden layers). Per-layer activation `alpha` values (LeakyReLU, ELU, CELU, PReLU) are extracted from the live torch module when available; PReLU per-channel slopes are honoured.

- `orbital()` now supports `mlp(engine = "nnet")` and direct `nnet::nnet()` models (single hidden layer, `skip = FALSE`). The handler validates `linout`, `softmax`, and `censored` against the requested mode and refuses unsupported combinations rather than silently miscompiling outputs.

- `orbital()` now supports `h2o::h2o.deeplearning()` models via the `agua` package. Hidden activations `Rectifier`/`RectifierWithDropout`, `Tanh`/`TanhWithDropout`, and `Maxout`/`MaxoutWithDropout` (with `maxout_size`) are supported. Multi-class class-order verification emits a warning when the H2O response domain cannot be read back from the model object. Multi-output regression is explicitly rejected.

## Known limitations

- **`MultiHeadAttention` and `use_causal_mask`**: In Keras 3, `use_causal_mask` is a call-time argument passed to `layer.__call__()` and is **not** stored in the saved layer config. As a result, `orbital()` cannot detect or enforce the causal mask at prediction time. If your model was trained with `use_causal_mask = TRUE`, the orbital translation will produce predictions that ignore the mask (attending to all positions, including future ones). A warning is emitted when `orbital()` encounters a `MultiHeadAttention` layer to draw attention to this limitation.

- **Keras `attention_mask` / recurrent masking**: orbital does not honour an explicit `attention_mask` argument passed to `MultiHeadAttention`/`GroupQueryAttention`/`Attention`/`AdditiveAttention`, nor does it honour upstream `Masking` / `mask_zero = TRUE` embeddings feeding `LSTM`/`GRU`/`SimpleRNN`. These configurations are refused with a clear error rather than silently ignored.

- **Keras `Softmax` axis**: only `axis = -1` (the last / feature axis) is supported. Other axis values are rejected at translation time because orbital's flat-column layout cannot express them.

# orbital 0.5.0

## New models

- `orbital()` now works with `boost_tree(engine = "catboost")` models for numeric, class, and probability predictions. (#90)

- `orbital()` now works with `boost_tree(engine = "lightgbm")` models for numeric, class, and probability predictions. (#89)

- `orbital()` now works with `decision_tree(engine = "rpart")` models for numeric, class, and probability predictions. (#128)

- `orbital()` now works with `mars(engine = "earth")` models for class and probability predictions. (#127)

- `orbital()` now works with `multinom_reg(engine = "glmnet")` models for class and probability predictions. (#127)

- `orbital()` now works with `rand_forest(engine = "randomForest")` models for class and probability predictions. (#127)

- `orbital()` now works with `rand_forest(engine = "ranger")` models for class and probability predictions. (#127)

## Improvements

- `orbital()` gains a `separate_trees` argument for tree ensemble models (xgboost, lightgbm, catboost, ranger, randomForest). When `TRUE`, each tree is emitted as a separate intermediate column before being summed, which can enable parallel evaluation in columnar databases like DuckDB, Snowflake, and BigQuery. For models with many trees, the final summation is automatically batched in groups of 50 to avoid expression depth limits in databases. See the "Separate trees" vignette for details. (#105)

- Added support for `step_spline_b()`, `step_spline_convex()`, `step_spline_monotone()`, `step_spline_natural()`, and `step_spline_nonnegative()` from the recipes package. (#99)

- `step_YeoJohnson()` is now supported. (#96)

- Binary classification probability predictions now generate cleaner code by having the second probability reference the first (e.g., `.pred_1 = 1 - .pred_0`) instead of duplicating the full expression. (#100)

- New "Database deployment" vignette shows how to deploy predictions to a database as tables or views. (#74)

- New "SQL size" vignette documents how model type and hyperparameters affect generated SQL size, and shows how to jointly tune for predictive performance and SQL complexity.

## Bug fixes

- All numeric values embedded in SQL expressions now use full IEEE 754 double precision (17 significant digits) to ensure exact round-trip accuracy between R and database predictions. This prevents subtle numerical drift in regularized model coefficients, normalized features, and tree split values. (#138)

# orbital 0.4.1

- Make work with new versions of xgboost. (#119)

# orbital 0.4.0

- Added support for tailor package and its integration into workflows. The following adjustments have gained `orbital()` support. (#103)
  - `adjust_equivocal_zone()`
  - `adjust_numeric_range()`
  - `adjust_predictions_custom()`
  - `adjust_probability_threshold()`

- Added `show_query()` method for orbital objects. (#106)

- Fixed printing bug where output would get malformed if coefficients had similarities. (#115)

# orbital 0.3.1

- Fixed bug where PCA steps didn't work if they were trained with more than 99 predictors. (#82)

- `step_pca_sparse()` no longer generate code with terms with 0 in them. (#51)

- Fixed bugs in all PCA steps where an error occurred depending on which predictors were selected. (#52)

- Fixed bug where large PCA results wouldn't work with data bases. (#84)

# orbital 0.3.0

- `orbital()` has gained `type` argument to change prediction type. (#66)

- `orbital()` now works with `logistic_reg(engine = "glm")` models for class prediction and probability predictions. (#62, #66)

- `orbital()` now works with `boost_tree(engine = "xgboost")` models for class prediction and probability predictions. (#71)

- `orbital()` now works with `decision_tree(engine = "partykit")` models for class prediction and probability predictions. (#77)

- `augment()` method for `orbital()` object have been added. (#55)

- `orbital()` gained `prefix` argument to allow for renaming of prediction columns. (#59)

# orbital 0.2.0

- Support for `step_dummy()`, `step_impute_mean()`, `step_impute_median()`, `step_impute_mode()`, `step_unknown()`, `step_novel()`, `step_other()`, `step_BoxCox()`, `step_inverse()`, `step_mutate()`, `step_sqrt()`, `step_indicate_na()`, `step_range()`, `step_intercept()`, `step_ratio()`, `step_lag()`, `step_log()`, `step_rename()` has been added. (#17)

- Support for `step_upsample()`, `step_smote()`, `step_smotenc()`, `step_bsmote()`, `step_adasyn()`, `step_rose()`, `step_downsample()`, `step_nearmiss()`, and `step_tomek()` has been added. (#21)

- Support for `step_bin2factor()`, `step_discretize()`, `step_lencode_mixed()`, `step_lencode_glm()`, `step_lencode_bayes()` has been added. (#22)

- Support for `step_pca_sparse()`, `step_pca_sparse_bayes()` and `step_pca_truncated()` as been added. (#23)

- `orbital()` now works on `tune::last_fit()` objects. (#13)

- `orbital_predict()` has been removed and replaced with the more idiomatic `predict()` method. (#10)

# orbital 0.1.0

- Initial CRAN submission.
