from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import numpy as np


class GraphExportError(RuntimeError):
    """Raised when graph export fails."""


def export_graph_for_gephi(
    graph: nx.Graph,
    *,
    output_dir: str = "exports",
    base_filename: str = "hypercube",
    projection: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, str]:
    """Export GraphML artifacts for Gephi and similar graph tools.

    When projection is provided, x/y/z are attached as node attributes so Gephi can
    use them directly in the Data Laboratory and styling pipelines.
    """
    if not isinstance(graph, nx.Graph):
        raise GraphExportError("graph must be a networkx.Graph instance.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    graphml_path = output_path / f"{base_filename}.graphml"

    export_graph = graph.copy()

    if projection is not None:
        for node in export_graph.nodes():
            if node in projection:
                coord = projection[node]
                export_graph.nodes[node]["x"] = float(coord[0])
                export_graph.nodes[node]["y"] = float(coord[1])
                export_graph.nodes[node]["z"] = float(coord[2])

    try:
        nx.write_graphml(export_graph, graphml_path)
    except Exception as exc:
        raise GraphExportError(f"Failed to write GraphML: {exc}") from exc

    return {
        "graphml_path": str(graphml_path.resolve()),
    }
