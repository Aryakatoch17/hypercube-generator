from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import networkx as nx
import numpy as np

from .core import binary_vertices_from_graph, generate_hypercube
from .graph_export import export_graph_for_gephi
from .isomorphism import check_graph_isomorphism
from .projection import project_hypercube_to_3d
from .visualization import save_hypercube_plots, visualize_hypercube
from .web_viewer import build_cytoscape_web_bundle


class OrchestrationError(RuntimeError):
    """Raised when the agentic orchestration workflow fails."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]


class HypercubeAgentOrchestrator:
    """Lightweight, deterministic agentic orchestrator for hypercube workflows."""

    def __init__(self) -> None:
        self.tools: Dict[str, ToolSpec] = {
            "generate_hypercube": ToolSpec(
                "generate_hypercube",
                "Generate an n-bit hypercube graph and adjacency matrix.",
                generate_hypercube,
            ),
            "project_hypercube_to_3d": ToolSpec(
                "project_hypercube_to_3d",
                "Project n-dimensional hypercube vertices into 3D.",
                project_hypercube_to_3d,
            ),
            "check_graph_isomorphism": ToolSpec(
                "check_graph_isomorphism",
                "Check isomorphism between two graphs using VF2.",
                check_graph_isomorphism,
            ),
            "save_hypercube_plots": ToolSpec(
                "save_hypercube_plots",
                "Save 3D projection plots as PNG and HTML.",
                save_hypercube_plots,
            ),
            "export_graph_for_gephi": ToolSpec(
                "export_graph_for_gephi",
                "Export GraphML for Gephi and graph analysis tools.",
                export_graph_for_gephi,
            ),
            "build_cytoscape_web_bundle": ToolSpec(
                "build_cytoscape_web_bundle",
                "Build Cytoscape.js web visualization bundle.",
                build_cytoscape_web_bundle,
            ),
            "visualize_hypercube": ToolSpec(
                "visualize_hypercube",
                "Generate and render an n-bit hypercube with PCA(3) and Plotly.",
                visualize_hypercube,
            ),
        }
        self.memory: Dict[str, Any] = {}

    def _parse_request(self, request: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(request, dict):
            if "n" not in request:
                raise OrchestrationError("Structured input must include key 'n'.")
            return {
                "n": int(request["n"]),
                "make_variant": bool(request.get("make_variant", True)),
                "project": bool(request.get("project", True)),
                "perspective": bool(request.get("perspective", False)),
                "create_plots": bool(request.get("create_plots", False)),
                "plot_output_dir": str(request.get("plot_output_dir", "plots")),
                "plot_max_nodes": int(request.get("plot_max_nodes", 512)),
                "export_graphml": bool(request.get("export_graphml", False)),
                "export_output_dir": str(request.get("export_output_dir", "exports")),
                "create_web_viewer": bool(request.get("create_web_viewer", False)),
                "web_max_nodes": int(request.get("web_max_nodes", 1200)),
                "create_plotly_pca_view": bool(request.get("create_plotly_pca_view", True)),
                "seed": int(request.get("seed", 42)),
            }

        if not isinstance(request, str) or not request.strip():
            raise OrchestrationError("Request must be a non-empty string or structured dict.")

        bit_match = re.search(r"(\d+)\s*[- ]?bit", request, flags=re.IGNORECASE)
        if not bit_match:
            raise OrchestrationError("Could not parse n-bit dimension from request.")

        lower = request.lower()
        make_variant = any(
            phrase in lower
            for phrase in ["variant", "permutation", "permute", "shuffle", "isomer"]
        )
        project = any(
            phrase in lower
            for phrase in ["project", "3d", "visual", "embed", "projection"]
        )
        perspective = "perspective" in lower
        create_plots = any(
            phrase in lower
            for phrase in ["plot", "visualize", "render", "html", "png", "image"]
        )
        export_graphml = any(
            phrase in lower
            for phrase in ["graphml", "gephi", "export", "graph file"]
        )
        create_web_viewer = any(
            phrase in lower
            for phrase in ["website", "browser", "web", "cytoscape", "same site"]
        )
        create_plotly_pca_view = any(
            phrase in lower
            for phrase in ["plotly", "pca", "3d graph", "interactive"]
        )
        sampled = any(phrase in lower for phrase in ["sample", "subset", "large", "16-bit"])

        return {
            "n": int(bit_match.group(1)),
            "make_variant": make_variant,
            "project": project,
            "perspective": perspective,
            "create_plots": create_plots,
            "plot_output_dir": "plots",
            "plot_max_nodes": 512 if sampled else 4096,
            "export_graphml": export_graphml,
            "export_output_dir": "exports",
            "create_web_viewer": create_web_viewer,
            "web_max_nodes": 1200 if sampled else 5000,
            "create_plotly_pca_view": create_plotly_pca_view,
            "seed": 42,
        }

    @staticmethod
    def _create_permuted_variant(graph: nx.Graph, seed: int = 42) -> Tuple[nx.Graph, Dict[str, str]]:
        labels = list(graph.nodes())
        rng = np.random.default_rng(seed)
        shuffled = list(np.array(labels)[rng.permutation(len(labels))])
        relabel_map = dict(zip(labels, shuffled))
        variant = nx.relabel_nodes(graph, relabel_map, copy=True)
        return variant, relabel_map

    def run(self, request: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Execute end-to-end agentic pipeline and return a human-readable summary."""
        parsed = self._parse_request(request)
        n = parsed["n"]
        seed = parsed["seed"]

        if n > 4:
            raise OrchestrationError(
                "This visualization-first pipeline supports n <= 4 for exact PCA + Plotly rendering."
            )

        try:
            return_adjacency = n <= 10
            graph, adjacency = self.tools["generate_hypercube"].fn(
                n,
                return_adjacency=return_adjacency,
            )
            self.memory["graph"] = graph
            self.memory["adjacency"] = adjacency

            projection = None
            plot_artifacts = None
            export_artifacts = None
            web_artifacts = None
            plotly_pca_artifacts = None
            if parsed["project"]:
                bit_vertices = binary_vertices_from_graph(graph)
                projection = self.tools["project_hypercube_to_3d"].fn(
                    bit_vertices,
                    list(graph.nodes()),
                    seed=seed,
                    perspective=parsed["perspective"],
                )
                self.memory["projection"] = projection

                if parsed["create_plotly_pca_view"]:
                    plotly_pca_artifacts = self.tools["visualize_hypercube"].fn(
                        n=n,
                        output_html=f"plots/hypercube_{n}bit_plotly_pca.html",
                    )
                    self.memory["plotly_pca_artifacts"] = plotly_pca_artifacts

                if parsed["create_plots"]:
                    plot_artifacts = self.tools["save_hypercube_plots"].fn(
                        graph,
                        projection,
                        output_dir=parsed["plot_output_dir"],
                        base_filename=f"hypercube_{n}bit",
                        max_plot_nodes=parsed["plot_max_nodes"],
                        seed=seed,
                    )
                    self.memory["plot_artifacts"] = plot_artifacts

            if parsed["export_graphml"]:
                export_artifacts = self.tools["export_graph_for_gephi"].fn(
                    graph,
                    output_dir=parsed["export_output_dir"],
                    base_filename=f"hypercube_{n}bit",
                    projection=projection,
                )
                self.memory["export_artifacts"] = export_artifacts

            if parsed["create_web_viewer"]:
                web_artifacts = self.tools["build_cytoscape_web_bundle"].fn(
                    graph,
                    output_dir=parsed["export_output_dir"],
                    base_filename=f"hypercube_{n}bit",
                    projection=projection,
                    max_web_nodes=parsed["web_max_nodes"],
                    seed=seed,
                )
                self.memory["web_artifacts"] = web_artifacts

            comparison_graph = graph
            relabel_map = None
            if parsed["make_variant"]:
                comparison_graph, relabel_map = self._create_permuted_variant(graph, seed=seed)
                self.memory["variant"] = comparison_graph
                self.memory["variant_relabel_map"] = relabel_map

            if comparison_graph is graph:
                iso_result = True
                iso_mapping = None
            else:
                iso_result, iso_mapping = self.tools["check_graph_isomorphism"].fn(
                    graph,
                    comparison_graph,
                )
            self.memory["isomorphic"] = iso_result
            self.memory["isomorphism_mapping"] = iso_mapping

        except Exception as exc:
            raise OrchestrationError(f"Workflow execution failed: {exc}") from exc

        vertex_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        projected_preview = None
        if projection:
            first_key = next(iter(projection))
            projected_preview = {
                "node": first_key,
                "coord": projection[first_key].round(6).tolist(),
            }

        summary = (
            f"Generated a {n}-bit hypercube with {vertex_count} vertices and {edge_count} edges. "
            f"Variant graph: {'yes' if parsed['make_variant'] else 'no'}. "
            f"3D projection: {'yes' if parsed['project'] else 'no'} "
            f"({'perspective' if parsed['perspective'] else 'orthographic'}). "
            f"Plots saved: {'yes' if parsed['create_plots'] and parsed['project'] else 'no'}. "
            f"GraphML exported: {'yes' if parsed['export_graphml'] else 'no'}. "
            f"Web viewer built: {'yes' if parsed['create_web_viewer'] else 'no'}. "
            f"Plotly PCA view: {'yes' if parsed['create_plotly_pca_view'] and parsed['project'] else 'no'}. "
            f"Isomorphic result (VF2): {iso_result}."
        )

        return {
            "summary": summary,
            "n": n,
            "vertex_count": vertex_count,
            "edge_count": edge_count,
            "adjacency_shape": adjacency.shape if adjacency is not None else None,
            "isomorphic": iso_result,
            "isomorphism_mapping": iso_mapping,
            "variant_relabel_map": relabel_map,
            "projection_preview": projected_preview,
            "plot_artifacts": plot_artifacts,
            "export_artifacts": export_artifacts,
            "web_artifacts": web_artifacts,
            "plotly_pca_artifacts": plotly_pca_artifacts,
        }
