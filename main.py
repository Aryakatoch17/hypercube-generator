from __future__ import annotations

import json

from hypercube_agentic.large_scale_pipeline import LargeScaleHypercubeAgent


def run_demo() -> None:
    agent = LargeScaleHypercubeAgent()
    result = agent.run("Generate a 16-bit hypercube process")

    print("=== Agentic Hypercube Pipeline Result ===")
    print(result["summary"])
    if result.get("heatmap_artifacts"):
        print("Heatmap files:")
        print(json.dumps(result["heatmap_artifacts"], indent=2))
    if result.get("spectral_result"):
        print("Spectral proof:")
        print(json.dumps(result["spectral_result"], indent=2))
    compact_result = {
        "n": result["n"],
        "node_count": result["node_count"],
        "edge_count": result["edge_count"],
        "num_variations": result["num_variations"],
        "base_nnz": result["base_nnz"],
        "heatmap_artifacts": result["heatmap_artifacts"],
        "spectral_result": result["spectral_result"],
    }
    print(json.dumps(compact_result, indent=2, default=str))


if __name__ == "__main__":
    run_demo()
