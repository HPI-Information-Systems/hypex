import logging
import pickle
import re
import typing as t
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.pyplot import contour

import hypaad

__all__ = ["ParameterModel"]


def networkx_to_json(graph: nx.DiGraph) -> t.Dict:
    return {"nodes": graph.nodes(), "edges": graph.edges()}


def json_to_networkx_to_json(data: t.Dict) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(data["nodes"])
    graph.add_edges_from(data["edges"])
    return graph


class ParameterModel:
    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        original_graph: nx.DiGraph,
        pruned_graph: nx.DiGraph,
        data_params: t.List[str],
        algorithm_params: t.List[str],
        param_models: t.Dict[str, "hypaad.NonLinearRegressionFrozenModel"],
        notes: t.Dict[str, t.Any] = {},
    ) -> None:
        self.original_graph = original_graph
        self.pruned_graph = pruned_graph
        self.data_params = data_params
        self.algorithm_params = algorithm_params
        self.param_models = param_models
        self.notes = notes

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "ParameterModel":
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def create_from(
        cls,
        graph: nx.DiGraph,
        data: pd.DataFrame,
        data_params: t.List[str],
        parameters: "hypaad.MultidimensionalParameterDistribution",
        edge_data: t.Dict[t.Tuple[int, int], "hypaad.RegressionResult"],
        score_variable: str = None,
    ) -> "ParameterModel":
        algorithm_params = list(set(graph.nodes) - set(data_params))
        pruned_graph = graph.copy()
        notes = {}

        if not nx.is_directed_acyclic_graph(graph):
            pruned_graph, removed_edges = hypaad.dfs_remove_back_edges(graph)
            notes["The graph must be a directed acyclic graph"] = [
                f"Removed edge {e} from graph" for e in removed_edges
            ]

        if not nx.is_directed_acyclic_graph(pruned_graph):
            cycle_edges = nx.find_cycle(pruned_graph, orientation="original")
            msg = f"The pruned graph is still not a directed acyclic graph. Edges forming a cycle: {cycle_edges}"
            notes["FATAL"] = [msg]
            cls._logger.error(msg)

            for u, v, direction in cycle_edges:
                pruned_graph.remove_edge(u, v)

        param_models: t.Dict[str, hypaad.NonLinearRegressionFrozenModel] = {}
        for node in algorithm_params:
            nodes_pre = list(pruned_graph.predecessors(node))
            # nodes_post = list(pruned_graph.successors(node))
            if len(nodes_pre) == 0:
                continue
                # if len(nodes_post) == 0:
                #     continue

                # def remove_all_successor_edges(start_node: str):
                #     removed_edges = []
                #     successors = list(pruned_graph.successors(start_node))
                #     for successor in successors:
                #         if (pruned_graph.predecessors(successor)) == 1:
                #             removed_edges.extend(
                #                 remove_all_successor_edges(successor)
                #             )
                #         pruned_graph.remove_edge(start_node, successor)
                #         removed_edges.append((start_node, successor))
                #     return removed_edges

                # removed_edges = remove_all_successor_edges(node)

                # notes[
                #     f"Algorithm parameter {node} has no predecessors, but has successors"
                # ] = [f"Removed edge {e} from graph" for e in removed_edges]
            else:
                data_x = data[nodes_pre]
                # data_x = pd.DataFrame(
                #     {
                #         node_pre: edge_data[(node_pre, node)].transform_funcs[node_pre](
                #             data[node_pre]
                #         )
                #         for node_pre in nodes_pre
                #     }
                # )

                data_y = np.array(data[node])
                sample_weight = data[score_variable] if score_variable else None

                parameter_distribution: hypaad.ParameterDistribution = [
                    dist
                    for dist in parameters.parameter_distributions
                    if dist.name == node
                ][0]
                result = hypaad.NonLinearRegression.fit(
                    data_x=data_x,
                    data_y=data_y,
                    sample_weight=sample_weight,
                    keep_dtype=True,
                    min_value=parameter_distribution.min_value,
                    max_value=parameter_distribution.max_value,
                )
                param_models[node] = result.model

        return cls(
            original_graph=graph,
            pruned_graph=pruned_graph,
            data_params=data_params,
            algorithm_params=algorithm_params,
            param_models=param_models,
            notes=notes,
        )

    def predict(self, **kwargs) -> t.Dict[str, t.Any]:
        missing_params = set(self.data_params) - kwargs.keys()
        if len(missing_params) > 0:
            raise ValueError(f"Missing data parameters {missing_params}")

        print("nodes: ", self.pruned_graph.nodes)
        print("edges: ", self.pruned_graph.edges)

        params = {}
        for node in nx.topological_sort(self.pruned_graph):
            nodes_pre = list(self.pruned_graph.predecessors(node))
            if len(nodes_pre) > 0:
                print("node:", node, "nodes_pre:", nodes_pre)
                params[node] = self.param_models[node].predict_single(
                    **kwargs, **params
                )

        return params

    def output_parameters(self) -> t.Set[str]:
        return set(self.param_models.keys())
