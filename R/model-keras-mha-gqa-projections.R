# GQA projection helper: shared Q/K/V linear-projection loop.
# Called by .k3_groupedqueryattention() in model-keras-mha-gqa.R.

# Build linear projection expressions for one set of query/key/value heads.
# Returns a list with:
#   $nm_arr   — character array of shape (T, n_heads, head_dim) with expr names
#   $nms      — character vector of generated expression names
#   $exprs    — character vector of generated expression strings
.gqa_linear_project <- function(
    T_seq, # integer: sequence length (T_q or T_kv)
    n_heads, # integer: number of heads (num_heads or num_kv)
    head_dim, # integer: head dimension
    C_in, # integer: input channel size
    in_at, # function(t, c) -> backtick'd input expression string
    w_at, # function(c, d, h) -> weight scalar as character
    b_at, # function(d, h) -> bias scalar as character
    lname, # string: layer name for naming
    tag # string: one of "Q", "K", "V" (used in generated name)
) {
    nm_arr <- array(NA_character_, dim = c(T_seq, n_heads, head_dim))
    out_nms <- character(0L)
    out_exprs <- character(0L)

    for (t in seq_len(T_seq)) {
        for (h in seq_len(n_heads)) {
            for (d in seq_len(head_dim)) {
                terms <- vapply(
                    seq_len(C_in),
                    function(c) {
                        paste0(
                            in_at(t, c),
                            " * ",
                            format_numeric(w_at(c, d, h))
                        )
                    },
                    character(1L)
                )
                nm <- paste0(
                    "orbital_gqa_",
                    lname,
                    "_",
                    tag,
                    "_t",
                    t,
                    "_h",
                    h,
                    "_d",
                    d
                )
                out_nms <- c(out_nms, nm)
                out_exprs <- c(
                    out_exprs,
                    paste0(
                        "(",
                        paste(terms, collapse = " + "),
                        " + ",
                        b_at(d, h),
                        ")"
                    )
                )
                nm_arr[t, h, d] <- nm
            }
        }
    }

    list(nm_arr = nm_arr, nms = out_nms, exprs = out_exprs)
}
