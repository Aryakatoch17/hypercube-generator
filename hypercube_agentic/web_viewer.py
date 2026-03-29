from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import numpy as np


class WebViewerBuildError(RuntimeError):
    """Raised when browser visualization artifacts cannot be created."""


def build_cytoscape_web_bundle(
    graph: nx.Graph,
    *,
    output_dir: str = "exports",
    base_filename: str = "hypercube",
    projection: Optional[Dict[str, np.ndarray]] = None,
    max_web_nodes: int = 1200,
    seed: int = 42,
) -> Dict[str, str | int | bool]:
    """Build a Cytoscape.js web visualization bundle from a graph.

    Produces two files:
    - JSON elements file (nodes/edges and optional projected positions)
    - HTML viewer that loads that JSON and renders interactively
    """
    if not isinstance(graph, nx.Graph):
        raise WebViewerBuildError("graph must be a networkx.Graph instance.")
    if max_web_nodes <= 0:
        raise WebViewerBuildError("max_web_nodes must be positive.")

    all_nodes = list(graph.nodes())
    sampled = len(all_nodes) > max_web_nodes

    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(all_nodes), size=max_web_nodes, replace=False)
        idx.sort()
        selected_nodes = [all_nodes[i] for i in idx]
        graph_to_render = graph.subgraph(selected_nodes).copy()
    else:
        selected_nodes = all_nodes
        graph_to_render = graph

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / f"{base_filename}_cy_elements.json"
    html_path = output_path / f"{base_filename}_cytoscape.html"

    node_elements = []
    for node in selected_nodes:
        data = {"id": str(node), "label": str(node)}
        if projection is not None and node in projection:
            coord = projection[node]
            data["x"] = float(coord[0])
            data["y"] = float(coord[1])
            data["z"] = float(coord[2])
        node_elements.append({"data": data})

    edge_elements = []
    for u, v in graph_to_render.edges():
        edge_elements.append(
            {
                "data": {
                    "id": f"{u}__{v}",
                    "source": str(u),
                    "target": str(v),
                }
            }
        )

    payload = {
        "meta": {
            "sampled": sampled,
            "seed": seed,
            "plotted_nodes": graph_to_render.number_of_nodes(),
            "plotted_edges": graph_to_render.number_of_edges(),
        },
        "elements": {
            "nodes": node_elements,
            "edges": edge_elements,
        },
    }

    try:
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        raise WebViewerBuildError(f"Failed to write Cytoscape JSON: {exc}") from exc

    payload_inline = json.dumps(payload)

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Hypercube Browser Viewer</title>
  <script src=\"https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js\"></script>
  <style>
    body {{ margin: 0; font-family: Segoe UI, sans-serif; background: #f4f5f8; color: #111; }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 12px; }}
    h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
    .meta {{ font-size: 13px; color: #444; margin-bottom: 10px; }}
    #cy {{ width: 100%; height: 82vh; background: white; border: 1px solid #d4d7df; border-radius: 10px; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Hypercube Viewer (Cytoscape.js)</h1>
    <div class=\"meta\" id=\"meta\">Loading graph...</div>
    <div id=\"cy\"></div>
  </div>
  <script>
    const payload = {payload_inline};

    async function start() {{
      const meta = payload.meta;
      document.getElementById('meta').textContent =
        `Nodes: ${{meta.plotted_nodes}} | Edges: ${{meta.plotted_edges}} | Sampled: ${{meta.sampled}}`;

      const hasProjection = payload.elements.nodes.length > 0 &&
        payload.elements.nodes[0].data.x !== undefined;

      const elements = [
        ...payload.elements.nodes,
        ...payload.elements.edges,
      ];

      const layout = hasProjection
        ? {{
            name: 'preset',
            positions: (n) => {{
              const d = n.data();
              return {{ x: d.x * 200, y: d.y * 200 }};
            }},
          }}
        : {{
            name: 'cose',
            animate: false,
            fit: true,
          }};

      cytoscape({{
        container: document.getElementById('cy'),
        elements,
        style: [
          {{
            selector: 'node',
            style: {{
              'background-color': '#d62728',
              'label': 'data(label)',
              'font-size': 8,
              'text-outline-width': 2,
              'text-outline-color': '#fff',
              'text-valign': 'top',
              'text-halign': 'right',
              'width': 9,
              'height': 9,
            }}
          }},
          {{
            selector: 'edge',
            style: {{
              'line-color': '#1f77b4',
              'curve-style': 'haystack',
              'width': 1,
              'opacity': 0.5,
            }}
          }}
        ],
        layout,
      }});
    }}

    start().catch((err) => {{
      document.getElementById('meta').textContent = 'Failed to load visualization data: ' + err;
    }});
  </script>
</body>
</html>
"""

    try:
        html_path.write_text(html, encoding="utf-8")
    except Exception as exc:
        raise WebViewerBuildError(f"Failed to write Cytoscape HTML: {exc}") from exc

    return {
        "web_html_path": str(html_path.resolve()),
        "web_json_path": str(json_path.resolve()),
        "sampled": sampled,
        "web_nodes": graph_to_render.number_of_nodes(),
        "web_edges": graph_to_render.number_of_edges(),
    }
