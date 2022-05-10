library(graph, quietly = TRUE)

process_graph <- function(graph, data, independence_test) {
    edges <- edges(graph)
    edge_list <- data.frame(matrix(0, ncol = 2, nrow = 0))
    colnames(edge_list) <- c("from_node", "to_node")

    for (node in names(edges)) {
        for (edge in edges[[node]]) {
            from_node <- colnames(data)[strtoi(node)]
            to_node <- colnames(data)[strtoi(edge)]
            edge_list[nrow(edge_list) + 1, ] = c(from_node, to_node)
        }
    }

    return(edge_list)
}
