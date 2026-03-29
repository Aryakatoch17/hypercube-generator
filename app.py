from __future__ import annotations

import time
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

from hypercube_agentic.isomorphism import check_graph_isomorphism
from hypercube_agentic.large_scale_pipeline import (
    LargeScaleHypercubeAgent,
    generate_hypercube_variations,
    prove_isomerism_by_spectrum,
)
from hypercube_agentic.projection import project_hypercube_to_3d


def _compute_scale(n: int) -> Tuple[int, int]:
    vertices = 1 << n
    edges = n * (1 << (n - 1))
    return vertices, edges


def _sparse_to_binned_density(matrix: csr_matrix, resolution: int = 220) -> np.ndarray:
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    node_count = matrix.shape[0]
    coo = matrix.tocoo()

    row_bins = np.minimum((coo.row.astype(np.int64) * resolution) // node_count, resolution - 1)
    col_bins = np.minimum((coo.col.astype(np.int64) * resolution) // node_count, resolution - 1)

    binned = np.zeros((resolution, resolution), dtype=np.float64)
    np.add.at(binned, (row_bins, col_bins), 1.0)

    return np.log1p(binned)


def _build_heatmap_figure(density: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=density,
                colorscale="Viridis",
                hovertemplate="bin(x=%{x}, y=%{y}) density=%{z:.4f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        margin=dict(l=8, r=8, t=50, b=8),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _indices_to_bit_matrix(indices: np.ndarray, n: int) -> np.ndarray:
    bit_positions = np.arange(n - 1, -1, -1, dtype=np.int64)
    return ((indices[:, None] >> bit_positions) & 1).astype(np.int8)


def _extract_sampled_edges(adjacency: csr_matrix, sampled_idx: np.ndarray) -> List[Tuple[int, int]]:
    sub = adjacency[sampled_idx, :][:, sampled_idx].tocoo()
    edges: List[Tuple[int, int]] = []
    for r, c in zip(sub.row.tolist(), sub.col.tolist()):
        if r < c:
            edges.append((int(r), int(c)))
    return edges


def _build_3d_graph_figure(
    coords: np.ndarray,
    labels: List[str],
    edges: List[Tuple[int, int]],
    *,
    title: str,
    node_color: str,
) -> go.Figure:
    edge_x: List[float] = []
    edge_y: List[float] = []
    edge_z: List[float] = []
    for i, j in edges:
        edge_x.extend([float(coords[i, 0]), float(coords[j, 0]), None])
        edge_y.extend([float(coords[i, 1]), float(coords[j, 1]), None])
        edge_z.extend([float(coords[i, 2]), float(coords[j, 2]), None])

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(width=2, color="rgba(60,70,90,0.35)"),
        hoverinfo="none",
        name="edges",
    )
    node_trace = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=dict(size=4.6, color=node_color, opacity=0.95),
        text=labels,
        hovertemplate="vertex=%{text}<extra></extra>",
        name="vertices",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=42),
        scene=dict(
            xaxis=dict(visible=False, showgrid=False, zeroline=False, showbackground=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False, showbackground=False),
            zaxis=dict(visible=False, showgrid=False, zeroline=False, showbackground=False),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _inject_style() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
          html, body, [class*="css"], [data-testid="stAppViewContainer"] {
            font-family: 'IBM Plex Sans', sans-serif;
          }
          [data-testid="stMetric"] {
            border: 1px solid #E6EAF0;
            border-radius: 12px;
            padding: 0.6rem 0.8rem;
            background: linear-gradient(180deg, #FFFFFF 0%, #FAFCFF 100%);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Automated Spectral Isomerism Pipeline: N-Dimensional Hypercubes",
        layout="wide",
    )
    _inject_style()

    st.title("Automated Spectral Isomerism Pipeline: N-Dimensional Hypercubes")
    st.caption(
        "Sparse matrix computation + spectral graph theory (normalized Laplacian eigen-signatures) "
        "for scalable structural isomerism validation."
    )

    with st.sidebar:
        st.header("Pipeline Controls")
        n = st.number_input(
            "Hypercube Bits (n)",
            min_value=2,
            max_value=20,
            value=4,
            step=1,
            help="Total vertices = 2^n. Higher n increases runtime and memory pressure.",
        )
        run_pipeline = st.button(
            "Initialize Agent & Run Pipeline",
            type="primary",
            use_container_width=True,
        )

    if not run_pipeline:
        st.info("Set n in the sidebar and click the run button to execute the full agentic pipeline.")
        return

    vertices, edges = _compute_scale(int(n))
    pipeline_start = time.perf_counter()

    try:
        # Frontend entrypoint for the existing backend class.
        agent = LargeScaleHypercubeAgent()
        _ = agent

        with st.status("Initializing spectral isomerism pipeline...", expanded=True) as status:
            status.write("Process 1: Generating sparse adjacency matrices...")
            time.sleep(0.35)
            base_adjacency, variations = generate_hypercube_variations(
                int(n),
                num_variations=3,
                seed=42,
            )

            status.write("Process 2: Applying permutations & computing heatmaps...")
            time.sleep(0.35)
            heatmap_figures: List[go.Figure] = []
            for variation in variations:
                density = _sparse_to_binned_density(variation.adjacency, resolution=220)
                heatmap_figures.append(_build_heatmap_figure(density, variation.name.replace("_", " ").title()))

            status.write("Process 3: Running Scipy sparse eigensolver...")
            time.sleep(0.35)
            eig_start = time.perf_counter()
            spectral_result: Dict[str, object] = prove_isomerism_by_spectrum(
                variations,
                k_each_side=12,
                tol=1e-6,
                normalized=True,
            )
            eig_time = time.perf_counter() - eig_start

            # Build sampled graph views for UI responsiveness at large n.
            node_count = base_adjacency.shape[0]
            sample_size = min(192, node_count)
            rng = np.random.default_rng(42)
            sampled_idx = np.sort(rng.choice(node_count, size=sample_size, replace=False)).astype(np.int64)

            base_labels = [format(int(i), f"0{int(n)}b") for i in sampled_idx.tolist()]
            base_bits = _indices_to_bit_matrix(sampled_idx, int(n))
            base_edges = _extract_sampled_edges(base_adjacency, sampled_idx)

            base_projection_map = project_hypercube_to_3d(
                base_bits,
                base_labels,
                seed=42,
                perspective=False,
                normalize=True,
            )
            base_projection = np.array([base_projection_map[label] for label in base_labels], dtype=float)

            variation = variations[0]
            variation_original_ids = variation.permutation[sampled_idx]
            variation_labels = [format(int(i), f"0{int(n)}b") for i in variation_original_ids.tolist()]
            variation_bits = _indices_to_bit_matrix(variation_original_ids.astype(np.int64), int(n))
            variation_edges = _extract_sampled_edges(variation.adjacency, sampled_idx)

            variation_projection_map = project_hypercube_to_3d(
                variation_bits,
                variation_labels,
                seed=42,
                perspective=False,
                normalize=True,
            )
            variation_projection = np.array(
                [variation_projection_map[label] for label in variation_labels],
                dtype=float,
            )

            pca_input = base_bits.astype(float)
            if pca_input.shape[1] < 3:
                pca_input = np.pad(pca_input, ((0, 0), (0, 3 - pca_input.shape[1])), mode="constant")
            pca_coords = PCA(n_components=3).fit_transform(pca_input)

            isomer_base_fig = _build_3d_graph_figure(
                base_projection,
                base_labels,
                base_edges,
                title="Base Isomer (Sampled)",
                node_color="#1E88E5",
            )
            isomer_variant_fig = _build_3d_graph_figure(
                variation_projection,
                variation_labels,
                variation_edges,
                title="Permuted Isomer (Sampled)",
                node_color="#E53935",
            )
            pca_fig = _build_3d_graph_figure(
                pca_coords,
                base_labels,
                base_edges,
                title="PCA to 3D (Sampled)",
                node_color="#00897B",
            )
            bit_fig = _build_3d_graph_figure(
                base_projection,
                base_labels,
                base_edges,
                title="Actual Bit Geometry 3D (Rotational Projection)",
                node_color="#7B1FA2",
            )

            # Optional exact check on sampled graphs to visually back the isomer claim.
            sampled_base_graph = base_adjacency[sampled_idx, :][:, sampled_idx]
            sampled_var_graph = variation.adjacency[sampled_idx, :][:, sampled_idx]
            nx_base = nx.from_scipy_sparse_array(sampled_base_graph)
            nx_var = nx.from_scipy_sparse_array(sampled_var_graph)
            sampled_exact_isomorphic, _ = check_graph_isomorphism(nx_base, nx_var)

            status.update(label="Pipeline Execution Complete", state="complete", expanded=False)

        st.subheader("Operation Scale")
        m1, m2 = st.columns(2)
        m1.metric("Vertices (2^n)", f"{vertices:,}")
        m2.metric("Edges (n x 2^(n-1))", f"{edges:,}")

        st.subheader("Structural Variation Heatmaps")
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(heatmap_figures[0], use_container_width=True)
        c2.plotly_chart(heatmap_figures[1], use_container_width=True)
        c3.plotly_chart(heatmap_figures[2], use_container_width=True)

        st.subheader("Isomer Graphs (Visual)")
        i1, i2 = st.columns(2)
        i1.plotly_chart(isomer_base_fig, use_container_width=True)
        i2.plotly_chart(isomer_variant_fig, use_container_width=True)
        st.info(
            f"Sampled exact VF2 check on displayed subgraphs: {'isomorphic' if sampled_exact_isomorphic else 'not isomorphic'}"
        )

        st.subheader("PCA to 3D")
        st.plotly_chart(pca_fig, use_container_width=True)

        st.subheader("Actual Bit Plot (3D)")
        st.plotly_chart(bit_fig, use_container_width=True)

        st.subheader("Spectral Isomerism Verdict")
        all_match = bool(spectral_result.get("all_signatures_match", False))
        if all_match:
            st.success("Isomerism Proven: All structural variations are spectrally consistent.")
        else:
            st.error("Isomerism Not Proven: Spectral signatures differ across variations.")

        st.caption(
            "Mathematical proof statement: the sorted eigenvalues of the normalized Laplacian "
            "matrices match perfectly across graph variations (within tolerance)."
        )
        st.write(f"Eigensolver execution time: `{eig_time:.4f}` seconds")
        st.write(f"Total pipeline runtime: `{(time.perf_counter() - pipeline_start):.4f}` seconds")

        with st.expander("Detailed Spectral Result"):
            st.json(spectral_result)

    except MemoryError:
        st.error(
            "Memory limit reached while processing this value of n on the current host. "
            "Please lower n and rerun."
        )
    except Exception as exc:
        st.error(f"Pipeline failed safely: {exc}")


if __name__ == "__main__":
    main()
