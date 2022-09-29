import logging
import typing as t
from dataclasses import dataclass
from itertools import combinations, permutations

import networkx as nx
import numpy as np
import pandas as pd

# pylint: disable=cyclic-import
import hypaad

from .independence_test import NonLinearIndependenceTest

__all__ = ["NonLinearPC"]


class NonLinearPC:

    _logger: logging.Logger
    edge_data: t.Dict[t.Tuple[str, str], "hypaad.RegressionResult"]

    @dataclass
    class Result:
        parameter_model: "hypaad.ParameterModel"
        graph_edges: pd.DataFrame

        def get_graph_hash(self):
            return nx.weisfeiler_lehman_graph_hash(
                self.parameter_model.pruned_graph
            )

    def __init__(
        self,
        edge_data: t.Dict[t.Tuple[str, str], "hypaad.RegressionResult"] = {},
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self.edge_data = edge_data.copy()

    @classmethod
    def _split_data_and_score(
        cls, data: pd.DataFrame, score_variable: str
    ) -> t.Tuple[pd.DataFrame, pd.Series]:
        if score_variable is not None:
            scores = data[score_variable] ** 2
            variables = list(set(data.columns) - set([score_variable]))
            data = data[variables]
        else:
            scores = None
            data = data
        return data, scores

    @classmethod
    def dry_run(
        cls, data: pd.DataFrame, score_variable: str = None
    ) -> t.Dict[t.Tuple[str, str], "hypaad.RegressionResult"]:
        edge_data: t.Dict[t.Tuple[str, str], hypaad.RegressionResult] = {}
        nodes = filter(lambda x: x != score_variable, list(data.columns))
        df, scores = cls._split_data_and_score(data, score_variable)
        for (node_i, node_j) in permutations(nodes, 2):
            edge_data[
                (node_i, node_j)
            ] = NonLinearIndependenceTest.independence_test(
                df, node_i, node_j, scores
            )
        return edge_data

    def estimate_skeleton(
        self,
        df: pd.DataFrame,
        alpha: float,
        beta: float,
        scores: np.array,
        fixed_edges: t.List[t.Tuple[str, str]] = [],
        **kwargs,
    ) -> t.Tuple:
        num_nodes = len(df.columns)
        nodes = list(df.columns)

        g = nx.complete_graph(num_nodes, nx.DiGraph())
        node_mapping = {idx: col for idx, col in enumerate(nodes)}
        g = nx.relabel_nodes(g, mapping=node_mapping)

        sep_set = {
            node_j: {node_i: set() for node_i in nodes} for node_j in nodes
        }

        def method_stable(kwargs):
            return ("method" in kwargs) and kwargs["method"] == "stable"

        l = 0
        while True:
            cont = False
            remove_edges = []
            for (node_i, node_j) in permutations(nodes, 2):
                if (node_i, node_j) in fixed_edges:
                    continue

                adj_i = list(g.neighbors(node_i))
                if node_j not in adj_i:
                    continue
                else:
                    adj_i.remove(node_j)
                if len(adj_i) >= l:
                    self._logger.debug("testing %s and %s", node_i, node_j)
                    self._logger.debug("neighbors of %s are %s", node_i, adj_i)
                    if len(adj_i) < l:
                        continue
                    for nodes_k in combinations(adj_i, l):
                        if len(nodes_k) == 0:
                            if (node_i, node_j) in self.edge_data.keys():
                                retval = self.edge_data[(node_i, node_j)]
                                self._logger.debug(
                                    "Using precomputed regression result for (%s, %s)",
                                    node_i,
                                    node_j,
                                )
                            else:
                                retval = (
                                    NonLinearIndependenceTest.independence_test(
                                        df, node_i, node_j, scores
                                    )
                                )
                                self.edge_data[(node_i, node_j)] = retval
                            is_independend = retval.score < alpha
                        else:

                            if not all(
                                [
                                    node_j in g.neighbors(node_k)
                                    for node_k in nodes_k
                                ]
                            ):
                                self._logger.debug(
                                    "Skipping conditional independence test of %s -> %s | %s as at least one node_k does not have an edge with %s",
                                    node_i,
                                    node_j,
                                    nodes_k,
                                    node_j,
                                )
                                continue

                            cols_k = set(nodes_k)
                            retval = NonLinearIndependenceTest.conditional_independence_test(
                                df, node_i, node_j, cols_k, scores
                            )
                            is_independend = False
                            is_independend = retval.score > beta

                        self._logger.debug(
                            "(%s, %s) | %s score is %s",
                            node_i,
                            node_j,
                            nodes_k,
                            retval.score,
                        )
                        if is_independend:
                            if g.has_edge(node_i, node_j):
                                self._logger.debug(
                                    "  -> remove edge (%s, %s)", node_i, node_j
                                )
                                if method_stable(kwargs):
                                    remove_edges.append((node_i, node_j))
                                else:
                                    g.remove_edge(node_i, node_j)
                            sep_set[node_i][node_j] |= set(nodes_k)
                            sep_set[node_j][node_i] |= set(nodes_k)
                            break
                    cont = True
            l += 1
            if method_stable(kwargs):
                g.remove_edges_from(remove_edges)
            if cont is False:
                break
            if ("max_reach" in kwargs) and (l > kwargs["max_reach"]):
                break

        return (g, sep_set)

    def estimate_cpdag(self, skel_graph, sep_set):
        """Estimate a CPDAG from the skeleton graph and separation sets
        returned by the estimate_skeleton() function.
        Args:
            skel_graph: A skeleton graph (an undirected networkx.Graph).
            sep_set: An 2D-array of separation set.
                The contents look like something like below.
                    sep_set[i][j] = set([k, l, m])
        Returns:
            An estimated DAG.
        """
        dag = skel_graph.to_directed()
        node_ids = skel_graph.nodes()
        for (i, j) in combinations(node_ids, 2):
            adj_i = set(dag.successors(i))
            if j in adj_i:
                continue
            adj_j = set(dag.successors(j))
            if i in adj_j:
                continue
            if sep_set[i][j] is None:
                continue
            common_k = adj_i & adj_j
            for k in common_k:
                if k not in sep_set[i][j]:
                    if dag.has_edge(k, i):
                        self._logger.debug("S: remove edge (%s, %s)", k, i)
                        dag.remove_edge(k, i)
                    if dag.has_edge(k, j):
                        self._logger.debug("S: remove edge (%s, %s)", k, j)
                        dag.remove_edge(k, j)

        def _has_both_edges(dag, i, j):
            return dag.has_edge(i, j) and dag.has_edge(j, i)

        def _has_any_edge(dag, i, j):
            return dag.has_edge(i, j) or dag.has_edge(j, i)

        def _has_one_edge(dag, i, j):
            return (
                (dag.has_edge(i, j) and (not dag.has_edge(j, i)))
                or (not dag.has_edge(i, j))
                and dag.has_edge(j, i)
            )

        def _has_no_edge(dag, i, j):
            return (not dag.has_edge(i, j)) and (not dag.has_edge(j, i))

        # For all the combination of nodes i and j, apply the following
        # rules.
        old_dag = dag.copy()
        while True:
            for (i, j) in permutations(node_ids, 2):
                # Rule 1: Orient i-j into i->j whenever there is an arrow k->i
                # such that k and j are nonadjacent.
                #
                # Check if i-j.
                if _has_both_edges(dag, i, j):
                    # Look all the predecessors of i.
                    for k in dag.predecessors(i):
                        # Skip if there is an arrow i->k.
                        if dag.has_edge(i, k):
                            continue
                        # Skip if k and j are adjacent.
                        if _has_any_edge(dag, k, j):
                            continue
                        # Make i-j into i->j
                        self._logger.debug("R1: remove edge (%s, %s)", j, i)
                        dag.remove_edge(j, i)
                        break

                # Rule 2: Orient i-j into i->j whenever there is a chain
                # i->k->j.
                #
                # Check if i-j.
                if _has_both_edges(dag, i, j):
                    # Find nodes k where k is i->k.
                    succs_i = set()
                    for k in dag.successors(i):
                        if not dag.has_edge(k, i):
                            succs_i.add(k)
                    # Find nodes j where j is k->j.
                    preds_j = set()
                    for k in dag.predecessors(j):
                        if not dag.has_edge(j, k):
                            preds_j.add(k)
                    # Check if there is any node k where i->k->j.
                    if len(succs_i & preds_j) > 0:
                        # Make i-j into i->j
                        self._logger.debug("R2: remove edge (%s, %s)", j, i)
                        dag.remove_edge(j, i)

                # Rule 3: Orient i-j into i->j whenever there are two chains
                # i-k->j and i-l->j such that k and l are nonadjacent.
                #
                # Check if i-j.
                if _has_both_edges(dag, i, j):
                    # Find nodes k where i-k.
                    adj_i = set()
                    for k in dag.successors(i):
                        if dag.has_edge(k, i):
                            adj_i.add(k)
                    # For all the pairs of nodes in adj_i,
                    for (k, l) in combinations(adj_i, 2):
                        # Skip if k and l are adjacent.
                        if _has_any_edge(dag, k, l):
                            continue
                        # Skip if not k->j.
                        if dag.has_edge(j, k) or (not dag.has_edge(k, j)):
                            continue
                        # Skip if not l->j.
                        if dag.has_edge(j, l) or (not dag.has_edge(l, j)):
                            continue
                        # Make i-j into i->j.
                        self._logger.debug("R3: remove edge (%s, %s)", j, i)
                        dag.remove_edge(j, i)
                        break

                # Rule 4: Orient i-j into i->j whenever there are two chains
                # i-k->l and k->l->j such that k and j are nonadjacent.
                #
                # However, this rule is not necessary when the PC-algorithm
                # is used to estimate a DAG.

            if nx.is_isomorphic(dag, old_dag):
                break
            old_dag = dag.copy()

        return dag

    # def orient_graph(self, df: pd.DataFrame, graph: nx.DiGraph):
    #     nodes = list(df.columns)
    #     g = graph.copy()

    #     for idx, node_i in enumerate(nodes):
    #         for node_j in nodes[idx + 1 :]:
    #             if not (
    #                 g.has_edge(node_i, node_j) and g.has_edge(node_j, node_i)
    #             ):
    #                 continue

    #             is_forward = (
    #                 self.edge_data[(node_i, node_j)]["score"]
    #                 > self.edge_data[(node_j, node_i)]["score"]
    #             )

    #             if is_forward:
    #                 g.remove_edge(node_j, node_i)
    #             else:
    #                 g.remove_edge(node_i, node_j)

    #     return g

    def run(
        self,
        data: pd.DataFrame,
        data_gen_vars: t.List[str],
        alpha: float,
        beta: float,
        parameters: "hypaad.MultidimensionalParameterDistribution",
        score_variable: str = None,
        verbose=False,
    ) -> "NonLinearPC.Result":
        self._logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        data_no_score, scores = self.__class__._split_data_and_score(
            data, score_variable
        )

        skeleton, sep_set = self.estimate_skeleton(
            df=data_no_score,
            alpha=alpha,
            beta=beta,
            scores=scores,
            verbose=verbose,
            fixed_edges=[],
        )

        fixed_predictor_vars = set(data_gen_vars)
        pruned_skeleton = skeleton.copy()
        for node_from, node_to in list(pruned_skeleton.edges):
            if node_to in fixed_predictor_vars:
                pruned_skeleton.remove_edge(node_from, node_to)

        nx_graph = self.estimate_cpdag(
            skel_graph=pruned_skeleton, sep_set=sep_set
        )

        parameter_model = hypaad.ParameterModel.create_from(
            graph=nx_graph,
            data=data,
            edge_data=self.edge_data,
            data_params=data_gen_vars,
            score_variable=score_variable,
            parameters=parameters,
        )

        graph_edges = pd.DataFrame.from_records(
            [
                {
                    "from": edge[0],
                    "to": edge[1],
                    **self.edge_data[edge].to_edge_data(),
                }
                for edge in nx_graph.edges
            ]
        )

        return NonLinearPC.Result(parameter_model, graph_edges)
