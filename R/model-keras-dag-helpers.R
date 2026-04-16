# Internal helper functions for DAG/Functional API layer parsing.
# Called by orbital_keras_dag_impl() in model-keras-dag.R.

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

# Internal: Masking layer pass-through handler.
# Masking metadata (the mask itself) cannot be expressed in pure SQL;
# expressions are forwarded unchanged to the next layer.
.k3_masking_passthru <- function(
  l,
  lname,
  topo_map,
  expr_reg,
  state,
  weight_map,
  output_layer_names,
  last_dense
) {
  inbound <- topo_map[[lname]]
  in_exprs <- get(inbound[1L], envir = expr_reg, inherits = FALSE)
  assign(lname, in_exprs, envir = expr_reg)
}
