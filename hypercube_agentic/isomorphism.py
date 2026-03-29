from __future__ import annotations

from typing import Dict, Optional, Tuple

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher


class IsomorphismCheckError(ValueError):
    """Raised when graph inputs are invalid for isomorphism checking."""


def check_graph_isomorphism(
    graph_a: nx.Graph,
    graph_b: nx.Graph,
) -> Tuple[bool, Optional[Dict[str, str]]]:
    """Check graph isomorphism using NetworkX VF2 algorithm.

    Args:
        graph_a: First graph.
        graph_b: Second graph.

    Returns:
        (is_isomorphic, mapping)
        - mapping is None if graphs are not isomorphic.

    Raises:
        IsomorphismCheckError: If inputs are not NetworkX Graph objects.
    """
    if not isinstance(graph_a, nx.Graph) or not isinstance(graph_b, nx.Graph):
        raise IsomorphismCheckError("Both inputs must be networkx.Graph objects.")

    matcher = GraphMatcher(graph_a, graph_b)
    isomorphic = matcher.is_isomorphic()
    mapping = dict(matcher.mapping) if isomorphic else None

    return isomorphic, mapping
