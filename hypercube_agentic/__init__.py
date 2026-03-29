"""Agentic hypercube generation and analysis package."""

from .core import generate_hypercube
from .projection import project_hypercube_to_3d
from .isomorphism import check_graph_isomorphism
from .orchestrator import HypercubeAgentOrchestrator
from .graph_export import export_graph_for_gephi
from .visualization import save_hypercube_plots, visualize_hypercube
from .web_viewer import build_cytoscape_web_bundle
from .large_scale_pipeline import (
    LargeScaleHypercubeAgent,
    generate_hypercube_variations,
    generate_sparse_hypercube_adjacency,
    prove_isomerism_by_spectrum,
    render_adjacency_heatmaps,
)

__all__ = [
    "generate_hypercube",
    "project_hypercube_to_3d",
    "check_graph_isomorphism",
    "HypercubeAgentOrchestrator",
    "export_graph_for_gephi",
    "save_hypercube_plots",
    "visualize_hypercube",
    "build_cytoscape_web_bundle",
    "generate_sparse_hypercube_adjacency",
    "generate_hypercube_variations",
    "render_adjacency_heatmaps",
    "prove_isomerism_by_spectrum",
    "LargeScaleHypercubeAgent",
]
