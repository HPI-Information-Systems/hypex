source("utils.R")

library(pcalg, quietly = TRUE)


cls_pc <- function(data, independence_test, alpha, cores, subset_size, skeleton_method,
    verbose) {
    indepTestDict <- list(gaussCI = gaussCItest, binCI = binCItest, disCI = disCItest)

    if (independence_test == "gaussCI") {
        matrix_df <- data.matrix(data)
        sufficient_stats <- list(C = cor(matrix_df), n = nrow(matrix_df))
    } else if (independence_test == "binCI" || independence_test == "disCI") {
        # Map categories to numbers if not done yet
        data[] <- lapply(data, factor)
        data <- data[sapply(data, function(x) !is.factor(x) | nlevels(x) > 1)]
        matrix_df <- data.matrix(data) - 1

        if (independence_test == "binCI") {
            sufficient_stats <- list(dm = matrix_df, adaptDF = FALSE)
        } else {
            p <- ncol(matrix_df)
            nlev <- vapply(seq_len(p), function(j) length(levels(factor(matrix_df[,
                j]))), 1L)
            sufficient_stats <- list(dm = matrix_df, adaptDF = FALSE, nlev = nlev)
        }
        # avoid segfaults in C++ extension, limit numCores to 1
        if (independence_test == "binCI") {
            cores <- 1
        }
    } else {
        stop("No valid independence test specified")
    }

    result = pc(suffStat = sufficient_stats, verbose = verbose, indepTest = indepTestDict[[independence_test]],
        m.max = subset_size, p = ncol(matrix_df), alpha = alpha, numCores = cores,
        skel.method = skeleton_method)

    return(process_graph(graph = result@graph, data = data, independence_test = independence_test))
}
