"""
Social interaction module.

The GridWorld environment enforces perceptual visibility rules directly
(fully_mixed / local / isolated). This module builds a NetworkX graph
representation of the same topology for analysis, logging and (optionally)
visualization -- e.g. to report average degree/density of the interaction
network at the end of an experiment.
"""
import networkx as nx
import numpy as np


def build_interaction_graph(positions, topology="fully_mixed", local_radius=3):
    n = len(positions)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if topology == "isolated":
        return G
    for i in range(n):
        for j in range(i + 1, n):
            if topology == "fully_mixed":
                G.add_edge(i, j)
            elif topology == "local":
                d = abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1])
                if d <= local_radius:
                    G.add_edge(i, j)
    return G


def topology_summary(positions, topology="fully_mixed", local_radius=3):
    G = build_interaction_graph(positions, topology, local_radius)
    n = G.number_of_nodes()
    degrees = dict(G.degree())
    avg_degree = float(np.mean(list(degrees.values()))) if n > 0 else 0.0
    return {
        "topology": topology,
        "num_edges": G.number_of_edges(),
        "avg_degree": avg_degree,
        "density": float(nx.density(G)) if n > 1 else 0.0,
    }
