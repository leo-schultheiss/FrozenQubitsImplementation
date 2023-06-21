import random

import networkx as nx
from Random_sparse_graph_generator import generate_randomwalk_graph, Graph
from helper_FrozenQubits import drop_hotspot_node
from helper_qaoa import pqc_QAOA


def convert_graph_classes(graph: Graph):
    new_graph = nx.Graph()
    edges = graph.edges
    nodes = graph.nodes
    new_graph.add_edges_from(edges)
    new_graph.add_nodes_from(nodes)
    return new_graph


def generate_2_regular(num_nodes) -> nx.Graph:
    nodes = list(range(num_nodes))
    G = Graph(nodes)
    for i in range(num_nodes):
        G.edges.append((i, (i + 1) % num_nodes))
    return convert_graph_classes(G)


def generate_random(num_nodes, num_edges):
    G = generate_randomwalk_graph(num_nodes, num_edges)
    G = convert_graph_classes(G)
    return G

def evaluate_cnot_difference(graph: nx.Graph, m):
    cnot_before = 2 * len(graph.edges)
    list_of_halting_qubits = []
    for i in range(m):
        graph, list_of_halting_qubits = drop_hotspot_node(G=graph, list_of_fixed_vars=list_of_halting_qubits, verbosity=0)
    cnot_after = 2 * len(graph.edges)
    return cnot_before, cnot_after


def generate_J_from_graph(graph: nx.Graph):
    J = dict()
    for edge in graph.edges:
        weight = random.choice([-1., 1.])
        J[edge] = weight
    return J


def generate_h_from_graph(graph: nx.Graph):
    h = dict()
    for node in graph.nodes:
        h[node] = 0.
    return h


def print_circ_from_graph(graph):
    J = generate_J_from_graph(graph)
    h = generate_h_from_graph(graph)
    _out = pqc_QAOA(J=J, h=h, num_layers=1)
    qaoa_circ = _out['qc']
    print(qaoa_circ)


if __name__ == "__main__":
    graph = generate_2_regular(3)
    print(graph.edges)
    print(f"before: %i, after: %i", evaluate_cnot_difference(graph, 1))
