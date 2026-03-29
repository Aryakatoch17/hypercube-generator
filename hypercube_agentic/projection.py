from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


class ProjectionError(ValueError):
    """Raised for projection-related input or numerical issues."""


def _validate_vertices(vertices: np.ndarray) -> np.ndarray:
    if not isinstance(vertices, np.ndarray):
        raise ProjectionError("vertices must be a NumPy array.")
    if vertices.ndim != 2:
        raise ProjectionError("vertices must be a 2D matrix of shape (num_vertices, n).")
    if vertices.shape[0] == 0 or vertices.shape[1] == 0:
        raise ProjectionError("vertices must be non-empty.")
    return vertices.astype(float, copy=False)


def _build_rotation_matrix(n: int, seed: Optional[int] = 42) -> np.ndarray:
    """Build a deterministic orthonormal rotation matrix in R^n via QR."""
    rng = np.random.default_rng(seed)
    random_matrix = rng.normal(size=(n, n))
    q, r = np.linalg.qr(random_matrix)

    # Normalize sign to make QR output deterministic for the same seed.
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q *= signs
    return q


def project_hypercube_to_3d(
    vertices: np.ndarray,
    node_labels: Iterable[str],
    *,
    seed: Optional[int] = 42,
    perspective: bool = False,
    camera_distance: float = 4.0,
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """Project n-dimensional binary vertices to 3D.

    Pipeline:
    1) Convert binary vertices {0,1}^n to Cartesian coordinates {-1,+1}^n.
    2) Apply an nD orthonormal rotation.
    3) Keep first 3 coordinates (orthographic projection) or apply perspective.

    Args:
        vertices: Binary vertex matrix of shape (V, n).
        node_labels: Node identifiers corresponding to vertices row order.
        seed: Random seed for deterministic rotation matrix.
        perspective: Whether to apply perspective scaling in x/y.
        camera_distance: Distance used by perspective projection.
        normalize: If True, scale projected points to unit max norm.

    Returns:
        Mapping of node label -> 3D coordinate vector.

    Raises:
        ProjectionError: For invalid input or unstable perspective geometry.
    """
    vertices = _validate_vertices(vertices)
    labels = list(node_labels)

    if len(labels) != vertices.shape[0]:
        raise ProjectionError(
            "node_labels length must match number of vertex rows in vertices."
        )

    n = vertices.shape[1]
    if n < 3:
        # Pad low-dimensional cubes to make an R^3 embedding explicit.
        vertices = np.pad(vertices, ((0, 0), (0, 3 - n)), mode="constant")
        n = 3

    # Binary-to-Cartesian mapping: 0 -> -1, 1 -> +1.
    nd_points = 2.0 * vertices - 1.0

    rotation = _build_rotation_matrix(n, seed=seed)
    rotated = nd_points @ rotation

    projected = rotated[:, :3].copy()

    if perspective:
        depth = camera_distance - projected[:, 2]
        if np.any(depth <= 1e-8):
            raise ProjectionError(
                "Perspective projection failed: camera_distance too small for geometry."
            )
        projected[:, 0] /= depth
        projected[:, 1] /= depth

    if normalize:
        max_norm = np.max(np.linalg.norm(projected, axis=1))
        if max_norm > 0:
            projected /= max_norm

    return {label: projected[idx] for idx, label in enumerate(labels)}
