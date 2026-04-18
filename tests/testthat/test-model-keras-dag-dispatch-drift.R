# Drift test: ensures the user-facing "supported layers" message inside
# .k3_dispatch_layer() (model-keras-dag-dispatch.R) lists every layer family
# the dispatcher actually handles via grepl() branches.
#
# This is the L5 audit-finding remediation: rather than auto-generate the list
# from a registry (which would lose the hand-curated readable formatting), we
# pin a smoke test that fails if a new grepl(...) branch is added without a
# corresponding entry in the abort message.

test_that(".k3_dispatch_layer abort message mentions every supported family", {
    dispatch_path <- testthat::test_path(
        "..",
        "..",
        "R",
        "model-keras-dag-dispatch.R"
    )
    skip_if(!file.exists(dispatch_path), "dispatcher source not found")

    src <- readLines(dispatch_path)

    # Extract the literal patterns from grepl("PATTERN", cls, ...) branches.
    # Filter out `!grepl(...)` lines first so exclusion sub-patterns (used to
    # disambiguate overlapping families) don't appear as bogus dispatch entries.
    pat_lines <- grep('grepl\\("[^"]+", cls', src, value = TRUE, perl = TRUE)
    pat_lines <- pat_lines[!grepl("!grepl", pat_lines, fixed = TRUE)]
    patterns <- regmatches(
        pat_lines,
        regexpr('(?<=grepl\\(")[^"]+', pat_lines, perl = TRUE)
    )
    patterns <- unique(patterns)

    # Strip regex word-boundary markers and escapes so we can match against
    # the human-readable enumeration in the abort message.
    family_keys <- gsub("\\\\b", "", patterns, fixed = TRUE)
    family_keys <- gsub("\\\\", "", family_keys, fixed = TRUE)
    family_keys <- tolower(family_keys)

    # Map dispatcher patterns -> a recognisable substring expected to appear
    # in the abort message (lower-cased).
    pat_to_msg_token <- c(
        "input" = "input",
        "dense" = "dense",
        "einsumdense" = "einsumdense",
        "embedding" = "embedding",
        "attention" = "attention",
        "multiheadattention" = "multiheadattention",
        "groupedqueryattention" = "groupedqueryattention",
        "additiveattention" = "additiveattention",
        "timedistributed" = "timedistributed",
        "normalization" = "normalization",
        "categoryencoding" = "categoryencoding",
        "integerlookup" = "integerlookup",
        "rescaling" = "rescaling",
        "lstm" = "lstm",
        "gru" = "gru",
        "simplernn" = "simplernn",
        "bidirectional" = "bidirectional",
        "conv1d" = "conv1d",
        "conv1dtranspose" = "conv1dtranspose",
        "depthwiseconv1d" = "depthwiseconv1d",
        "separableconv1d" = "separableconv1d",
        "convlstm" = "convlstm",
        "convlstm1d" = "convlstm1d",
        "softmax" = "softmax",
        "activation" = "activation",
        "subtract" = "subtract",
        "concatenate" = "concatenate",
        "add" = "add",
        "flatten" = "flatten",
        "reshape" = "reshape",
        "dropout" = "dropout",
        "masking" = "masking",
        "batchnorm" = "batchnorm",
        "batchnormalization" = "batchnormalization",
        "layernorm" = "layernorm",
        "layernormalization" = "layernormalization",
        "groupnorm" = "groupnorm",
        "groupnormalization" = "groupnormalization",
        "rmsnormalization" = "rmsnormalization",
        "instancenorm" = "instancenorm",
        "instancenormalization" = "instancenormalization",
        "unitnorm" = "unitnorm",
        "unitnormalization" = "unitnormalization",
        "zeropadding" = "zeropadding",
        "zeropadding1d" = "zeropadding1d",
        "globalaveragepool" = "globalaveragepool",
        "globalaveragepooling" = "globalaveragepooling",
        "globalmaxpool" = "globalmaxpool",
        "globalmaxpooling" = "globalmaxpooling",
        "globalsumpooling" = "globalsumpooling",
        "averagepooling" = "averagepooling",
        "averagepooling1d" = "averagepooling1d",
        "maxpooling" = "maxpooling",
        "maxpooling1d" = "maxpooling1d",
        "adaptiveaveragepooling" = "adaptiveaveragepooling",
        "adaptiveaveragepooling1d" = "adaptiveaveragepooling1d",
        "adaptivemaxpooling" = "adaptivemaxpooling",
        "adaptivemaxpooling1d" = "adaptivemaxpooling1d",
        "upsampling" = "upsampling",
        "upsampling1d" = "upsampling1d",
        "permute" = "permute",
        "cropping1d" = "cropping1d",
        "repeatvector" = "repeatvector",
        "prelu" = "prelu",
        "leakyrelu" = "leakyrelu",
        "elu" = "elu",
        "relu" = "relu",
        "average" = "average",
        "maximum" = "maximum",
        "minimum" = "minimum",
        "multiply" = "multiply",
        "dot" = "dot"
    )

    # Locate the abort message body (between the first cli_abort( after the
    # final-else and its closing paren).
    msg_blob <- tolower(paste(src, collapse = " "))
    abort_marker <- "unsupported layer type in keras functional model"
    abort_idx <- regexpr(abort_marker, msg_blob, fixed = TRUE)
    expect_true(
        abort_idx > 0L,
        info = "abort marker missing from dispatcher source"
    )
    msg_after <- substr(msg_blob, abort_idx, abort_idx + 4000L)

    # For every grepl pattern we know how to map, confirm the message mentions
    # the corresponding family token. Patterns we don't know about are
    # collected and surfaced as a single failure to make the next maintainer
    # extend `pat_to_msg_token` deliberately.
    unmapped <- setdiff(family_keys, names(pat_to_msg_token))
    expect_equal(
        unmapped,
        character(0L),
        info = paste0(
            "Unmapped grepl patterns in dispatcher: ",
            paste(unmapped, collapse = ", "),
            ". Add them to `pat_to_msg_token` in this test, then update the ",
            "user-facing supported-layers message in model-keras-dag-dispatch.R."
        )
    )

    for (key in family_keys) {
        token <- pat_to_msg_token[[key]]
        expect_true(
            grepl(token, msg_after, fixed = TRUE),
            info = paste0(
                "Dispatcher handles `",
                key,
                "` via grepl() but the user-facing supported-layers message ",
                "does not mention `",
                token,
                "`. Update the message in model-keras-dag-dispatch.R."
            )
        )
    }
})
