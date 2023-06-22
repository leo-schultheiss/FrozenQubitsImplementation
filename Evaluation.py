import random

import networkx as nx
import numpy as np

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


def generate_random(num_nodes, num_edges=-1):
    if num_edges == -1:
        num_edges = num_nodes - 1
    G = generate_randomwalk_graph(num_nodes, num_edges)
    G = convert_graph_classes(G)
    return G


def evaluate_cnot_count_for_m(graph: nx.Graph, m):
    list_of_halting_qubits = []
    for i in range(m):
        graph, list_of_halting_qubits = drop_hotspot_node(G=graph, list_of_fixed_vars=list_of_halting_qubits,
                                                          verbosity=0)
    return get_cnot_count(graph)


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


def get_cnot_count(graph: nx.Graph):
    return len(graph.edges) * 2


# todo
# evaluate cnot diff for n = 4..24, over m = 0,1,2
def run_evaluation_random_walk(additional_edges=0):
    if additional_edges < 0:
        raise TypeError("additional edges must be positive")
    result = []
    for n in range(4, 25):
        graph = generate_random(n, n - 1 + additional_edges)
        current_m = []
        # this will also generate the baseline
        for m in range(3):
            cnot_m = evaluate_cnot_count_for_m(graph, m)
            current_m.append(cnot_m)
        result.append(current_m)
    return result


def run_evaluation_2_regular():
    result = []
    for n in range(4, 25):
        graph = generate_2_regular(n)
        current_m = []
        # this will also generate the baseline
        for m in range(3):
            cnot_m = evaluate_cnot_count_for_m(graph, m)
            current_m.append(cnot_m)
        result.append(current_m)
    return result


def run_average_random(n: int, additional_edges=0):
    results = []
    for i in range(n):
        results.append(run_evaluation_random_walk(additional_edges))
    return results


def run_average_2_regular(n: int):
    results = []
    for i in range(n):
        results.append(run_evaluation_2_regular())
    return results


def average_results(tensor: [[[int]]]):
    tensor = np.array(tensor)
    # Calculate the average along each axis
    averages = np.mean(tensor, axis=0)

    return averages


def print_matrix_as_csv(averages: [[float]]):
    matrix = np.array(averages)
    # Get the number of rows and columns in the matrix
    num_rows, num_cols = matrix.shape

    # Loop through each row of the matrix
    for i in range(num_rows):
        # Calculate the index+4 value for the current row
        index_value = i + 4

        # Create a list to store the values in the current row
        row_values = []

        # Loop through each column of the matrix
        for j in range(num_cols):
            # Append the value to the row_values list
            row_values.append(str(matrix[i, j]))

        # Concatenate the index+4 value and the row values into a CSV string
        csv_string = str(index_value) + ', ' + ', '.join(row_values)

        # Print the CSV string
        print(csv_string)


if __name__ == "__main__":
    print(generate_random(6, 5).edges)
    random.seed(12345678910)
    n = 5
    results_2_regular = run_average_2_regular(n)
    averages_2_regular = average_results(results_2_regular)
    print("2-regular csv:")
    print_matrix_as_csv(averages_2_regular)

    for additional_edges in range(2):
        results_random = run_average_random(n, additional_edges)
        averages_random = average_results(results_random)
        print("random with edge + " + str(additional_edges))
        print_matrix_as_csv(averages_random)
