from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import eigsh


class LargeScalePipelineError(RuntimeError):
    """Raised when large-scale hypercube pipeline steps fail."""


@dataclass(frozen=True)
class HypercubeVariation:
    """Container for a permuted hypercube variation."""

    name: str
    adjacency: csr_matrix
    permutation: np.ndarray


def _validate_n(n: int) -> None:
    if not isinstance(n, int):
        raise LargeScalePipelineError("n must be an integer.")
    if n <= 0:
        raise LargeScalePipelineError("n must be positive.")
    if n > 22:
        raise LargeScalePipelineError("n is too large for this workstation-oriented pipeline (max: 22).")


def generate_sparse_hypercube_adjacency(n: int) -> csr_matrix:
    """Generate sparse adjacency matrix of n-dimensional hypercube.

    Recurrence:
        A_1 = [[0,1],[1,0]]
        A_n = kron(A_{n-1}, I_2) + kron(I_{2^(n-1)}, A_1)
    """
    _validate_n(n)

    a1 = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))
    adjacency = a1

    for dim in range(2, n + 1):
        i2 = eye(2, format="csr", dtype=np.float64)
        i_prev = eye(1 << (dim - 1), format="csr", dtype=np.float64)
        adjacency = kron(adjacency, i2, format="csr") + kron(i_prev, a1, format="csr")

    return adjacency.tocsr()


def generate_hypercube_variations(
    n: int,
    *,
    num_variations: int = 3,
    seed: int = 42,
) -> Tuple[csr_matrix, List[HypercubeVariation]]:
    """Generate base sparse hypercube and permuted structural variations."""
    if num_variations <= 0:
        raise LargeScalePipelineError("num_variations must be positive.")

    base = generate_sparse_hypercube_adjacency(n)
    node_count = base.shape[0]
    rng = np.random.default_rng(seed)

    variations: List[HypercubeVariation] = []
    for idx in range(num_variations):
        perm = rng.permutation(node_count)
        permuted = base[perm, :][:, perm].tocsr()
        variations.append(
            HypercubeVariation(
                name=f"variation_{idx + 1}",
                adjacency=permuted,
                permutation=perm,
            )
        )

    return base, variations


def _sparse_to_binned_density(matrix: csr_matrix, resolution: int = 512) -> np.ndarray:
    """Downsample sparse adjacency into a dense heatmap-friendly bin matrix."""
    if resolution <= 0:
        raise LargeScalePipelineError("resolution must be positive.")

    n = matrix.shape[0]
    coo = matrix.tocoo()

    row_bins = np.minimum((coo.row.astype(np.int64) * resolution) // n, resolution - 1)
    col_bins = np.minimum((coo.col.astype(np.int64) * resolution) // n, resolution - 1)

    binned = np.zeros((resolution, resolution), dtype=np.float64)
    np.add.at(binned, (row_bins, col_bins), 1.0)

    # Log scale preserves structure across very large n.
    return np.log1p(binned)


def render_adjacency_heatmaps(
    variations: Sequence[HypercubeVariation],
    *,
    output_html: str = "plots/hypercube_variations_heatmap.html",
    resolution: int = 512,
) -> Dict[str, object]:
    """Render adjacency matrix heatmaps for structural variations."""
    if len(variations) == 0:
        raise LargeScalePipelineError("At least one variation is required for heatmap rendering.")

    heatmaps = [_sparse_to_binned_density(v.adjacency, resolution=resolution) for v in variations]

    fig = make_subplots(
        rows=1,
        cols=len(variations),
        subplot_titles=[v.name for v in variations],
        horizontal_spacing=0.03,
    )

    for i, heat in enumerate(heatmaps, start=1):
        fig.add_trace(
            go.Heatmap(
                z=heat,
                colorscale="Viridis",
                showscale=(i == len(variations)),
                hovertemplate="bin(x=%{x}, y=%{y}) density=%{z:.4f}<extra></extra>",
            ),
            row=1,
            col=i,
        )

    fig.update_layout(
        title="Hypercube Structural Variations: Sparse Adjacency Heatmaps",
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    for i in range(1, len(variations) + 1):
        fig.update_xaxes(visible=False, row=1, col=i)
        fig.update_yaxes(visible=False, row=1, col=i)

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")

    return {
        "heatmap_html_path": str(output_path.resolve()),
        "heatmap_resolution": resolution,
        "figures": [v.name for v in variations],
    }


def spectral_signature(
    adjacency: csr_matrix,
    *,
    k_each_side: int = 12,
    normalized: bool = True,
) -> np.ndarray:
    """Compute fast spectral signature using sparse eigensolvers.

    For large sparse graphs, a full eigendecomposition is intractable. This computes
    a stable signature using extremal eigenvalues from both sides of the spectrum.
    """
    if k_each_side <= 0:
        raise LargeScalePipelineError("k_each_side must be positive.")

    matrix = adjacency
    if normalized:
        matrix = sparse_laplacian(adjacency, normed=True).asfptype().tocsr()

    n = matrix.shape[0]
    if n <= 3:
        dense = matrix.toarray()
        vals = np.linalg.eigvalsh(dense)
        return np.sort(vals)

    k = min(k_each_side, n - 2)
    eig_kwargs = {
        "return_eigenvectors": False,
        "tol": 1e-7,
        "maxiter": max(5000, n // 4),
        "ncv": min(n - 1, max(2 * k + 8, 24)),
    }

    largest = eigsh(matrix, k=k, which="LA", **eig_kwargs)
    smallest = eigsh(matrix, k=k, which="SA", **eig_kwargs)

    signature = np.sort(np.concatenate([smallest, largest]))
    return signature


def prove_isomerism_by_spectrum(
    variations: Sequence[HypercubeVariation],
    *,
    k_each_side: int = 12,
    tol: float = 1e-6,
    normalized: bool = True,
) -> Dict[str, object]:
    """Compare spectral signatures across variations and conclude isomerism."""
    if len(variations) < 2:
        raise LargeScalePipelineError("At least two variations are required for comparison.")

    signatures = {
        v.name: spectral_signature(
            v.adjacency,
            k_each_side=k_each_side,
            normalized=normalized,
        )
        for v in variations
    }

    names = list(signatures.keys())
    base = signatures[names[0]]
    pairwise: Dict[str, bool] = {}

    for name in names[1:]:
        pairwise[f"{names[0]}_vs_{name}"] = bool(np.allclose(base, signatures[name], atol=tol, rtol=0.0))

    all_match = all(pairwise.values())
    verdict = "Yes, Isomerism Exists" if all_match else "No, Isomerism Not Proven"

    return {
        "verdict": verdict,
        "all_signatures_match": all_match,
        "pairwise_matches": pairwise,
        "signature_length": int(base.shape[0]),
        "tolerance": tol,
        "normalized_laplacian": normalized,
    }


class LargeScaleHypercubeAgent:
    """Agentic orchestrator for large-N sparse hypercube processing."""

    def _parse_request(self, request: Union[str, Dict[str, object]]) -> Dict[str, object]:
        if isinstance(request, dict):
            if "n" not in request:
                raise LargeScalePipelineError("Structured request must include key 'n'.")
            return {
                "n": int(request["n"]),
                "num_variations": int(request.get("num_variations", 3)),
                "seed": int(request.get("seed", 42)),
                "heatmap_resolution": int(request.get("heatmap_resolution", 512)),
                "k_each_side": int(request.get("k_each_side", 12)),
                "tol": float(request.get("tol", 1e-6)),
            }

        if not isinstance(request, str) or not request.strip():
            raise LargeScalePipelineError("Request must be a non-empty string or a structured dictionary.")

        match = re.search(r"(\d+)\s*[- ]?(bit|dimension|dimensional)", request, flags=re.IGNORECASE)
        if not match:
            match = re.search(r"(\d+)", request)
        if not match:
            raise LargeScalePipelineError("Could not parse n from request.")

        n = int(match.group(1))
        return {
            "n": n,
            "num_variations": 3,
            "seed": 42,
            "heatmap_resolution": 512,
            "k_each_side": 12,
            "tol": 1e-6,
        }

    def run(self, request: Union[str, Dict[str, object]]) -> Dict[str, object]:
        params = self._parse_request(request)

        n = int(params["n"])
        num_variations = int(params["num_variations"])
        seed = int(params["seed"])
        heatmap_resolution = int(params["heatmap_resolution"])
        k_each_side = int(params["k_each_side"])
        tol = float(params["tol"])

        base, variations = generate_hypercube_variations(
            n,
            num_variations=num_variations,
            seed=seed,
        )

        heatmap_artifacts = render_adjacency_heatmaps(
            variations,
            output_html=f"plots/hypercube_{n}bit_variations_heatmap.html",
            resolution=heatmap_resolution,
        )

        spectral_result = prove_isomerism_by_spectrum(
            variations,
            k_each_side=k_each_side,
            tol=tol,
            normalized=True,
        )

        summary = (
            f"Generated sparse {n}-dimensional hypercube with {base.shape[0]} nodes and {base.nnz // 2} edges. "
            f"Built {num_variations} permuted structural variations. "
            f"Rendered adjacency heatmaps. Spectral verdict: {spectral_result['verdict']}."
        )

        return {
            "summary": summary,
            "n": n,
            "node_count": int(base.shape[0]),
            "edge_count": int(base.nnz // 2),
            "num_variations": num_variations,
            "base_nnz": int(base.nnz),
            "heatmap_artifacts": heatmap_artifacts,
            "spectral_result": spectral_result,
        }
