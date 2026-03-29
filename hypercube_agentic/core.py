from __future__ import annotations

from typing import List, Optional, Tuple

import networkx as nx
import numpy as np


class HypercubeGenerationError(ValueError):
    """Raised when hypercube generation input is invalid."""


def _validate_dimension(n: int) -> None:
    if not isinstance(n, int):
        raise HypercubeGenerationError("n must be an integer.")
    if n <= 0:
        raise HypercubeGenerationError("n must be positive.")
    if n > 20:
        raise HypercubeGenerationError(
            "n is too large for in-memory generation in this implementation (max: 20)."
        )


def _binary_vertex_labels(n: int) -> List[str]:
    vertex_count = 1 << n
    return [format(i, f"0{n}b") for i in range(vertex_count)]


def _binary_vertex_matrix(n: int) -> np.ndarray:
    vertex_count = 1 << n
    ids = np.arange(vertex_count, dtype=np.uint64)
    bit_positions = np.arange(n - 1, -1, -1, dtype=np.uint64)
    return ((ids[:, None] >> bit_positions) & 1).astype(np.int8)


def generate_hypercube(
    n: int,
    *,
    return_adjacency: bool = True,
    max_dense_adjacency_n: int = 10,
) -> Tuple[nx.Graph, Optional[np.ndarray]]:
    """Generate the n-dimensional hypercube graph and its adjacency matrix.

    A vertex is represented by its n-bit binary string. Two vertices are
    connected iff their Hamming distance is exactly 1.

    Args:
        n: Number of bits / dimensions.

    Returns:
        A tuple of:
        - networkx.Graph with binary-string node labels
        - adjacency matrix as a NumPy array (shape: [2**n, 2**n]) or None

    Raises:
        HypercubeGenerationError: If n is invalid.
    """
    _validate_dimension(n)

    labels = _binary_vertex_labels(n)
    graph = nx.Graph()
    graph.add_nodes_from(labels)

    for i in range(1 << n):
        source = labels[i]
        for bit in range(n):
            neighbor_idx = i ^ (1 << bit)
            if i < neighbor_idx:
                graph.add_edge(source, labels[neighbor_idx])

    adjacency: Optional[np.ndarray] = None
    if return_adjacency:
        if n > max_dense_adjacency_n:
            raise HypercubeGenerationError(
                "Dense adjacency matrix is too large for this n. "
                "Use return_adjacency=False for high-dimensional cubes."
            )
        adjacency = nx.to_numpy_array(graph, nodelist=labels, dtype=np.int8)

    return graph, adjacency


def binary_vertices_from_graph(graph: nx.Graph) -> np.ndarray:
    """Extract binary vertices from graph node labels into a 0/1 matrix."""
    if not graph.nodes:
        raise HypercubeGenerationError("Graph is empty.")

    labels = list(graph.nodes)
    n = len(labels[0])
    if any(len(label) != n for label in labels):
        raise HypercubeGenerationError("Node labels are not uniform-length bit strings.")

    try:
        bit_matrix = np.array([[int(ch) for ch in label] for label in labels], dtype=np.int8)
    except ValueError as exc:
        raise HypercubeGenerationError(
            "Node labels must contain only binary digits (0 or 1)."
        ) from exc

    return bit_matrix
