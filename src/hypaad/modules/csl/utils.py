import logging
import typing as t

import networkx as nx
import numpy as np

__all__ = ["dfs_remove_back_edges"]

_logger = logging.getLogger(__name__)


def dfs_visit_recursively(
    graph: nx.DiGraph,
    node,
    nodes_color: t.Dict[str, int],
    edges_to_be_removed: t.List[t.Tuple[str, str]],
):

    nodes_color[node] = 1
    nodes_order = list(graph.successors(node))
    nodes_order = np.random.permutation(nodes_order)
    for child in nodes_order:
        if nodes_color[child] == 0:
            dfs_visit_recursively(graph, child, nodes_color, edges_to_be_removed)
        elif nodes_color[child] == 1:
            edges_to_be_removed.append((node, child))

    nodes_color[node] = 2


def dfs_remove_back_edges(
    graph: nx.DiGraph,
) -> t.Tuple[nx.DiGraph, t.List[t.Tuple[str, str]]]:
    """
    0: white, not visited
    1: grey, being visited
    2: black, already visited
    """

    nodes_color = {}
    edges_to_be_removed = []
    for node in graph.nodes():
        nodes_color[node] = 0

    nodes_order = list(graph.nodes())
    nodes_order = np.random.permutation(nodes_order)
    num_dfs = 0
    for node in nodes_order:
        if nodes_color[node] == 0:
            num_dfs += 1
            dfs_visit_recursively(graph, node, nodes_color, edges_to_be_removed)

    _logger.info("number of nodes to start dfs: %d", num_dfs)
    _logger.info("number of back edges: %d", len(edges_to_be_removed))

    pruned_graph = graph.copy()
    for edge in edges_to_be_removed:
        pruned_graph.remove_edge(edge[0], edge[1])
    return pruned_graph, edges_to_be_removed
