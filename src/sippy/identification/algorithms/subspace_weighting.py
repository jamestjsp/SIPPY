from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubspaceWeightingDiagnostics:
    requested: str
    applied: str
    covariance_rank: int
    covariance_rows: int
    fallback_reason: str | None = None


def cva_weighted_svd(consistent_subspace, conditional_outputs):
    consistent = np.asarray(consistent_subspace, dtype=float)
    conditional = np.asarray(conditional_outputs, dtype=float)
    if consistent.ndim != 2 or conditional.ndim != 2:
        raise ValueError("CVA inputs must be matrices")
    if consistent.shape[0] != conditional.shape[0]:
        raise ValueError(
            "consistent subspace and conditional outputs must have equal row counts"
        )

    covariance_product = conditional @ conditional.T
    covariance = 0.5 * (covariance_product + covariance_product.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest_eigenvalue = max(float(eigenvalues[-1]), 0.0)
    tolerance = max(covariance.shape) * np.finfo(np.float64).eps * largest_eigenvalue
    retained = eigenvalues > tolerance
    covariance_rank = int(np.count_nonzero(retained))
    covariance_rows = covariance.shape[0]
    if covariance_rank < covariance_rows:
        decomposition = np.linalg.svd(consistent, full_matrices=False)
        diagnostics = SubspaceWeightingDiagnostics(
            requested="CVA",
            applied="unweighted",
            covariance_rank=covariance_rank,
            covariance_rows=covariance_rows,
            fallback_reason="conditional_covariance_rank_deficient",
        )
        return (*decomposition, None, diagnostics)

    square_roots = np.sqrt(eigenvalues)
    square_root = (eigenvectors * square_roots) @ eigenvectors.T
    inverse_square_root = (eigenvectors * (1.0 / square_roots)) @ eigenvectors.T
    decomposition = np.linalg.svd(
        inverse_square_root @ consistent,
        full_matrices=False,
    )
    diagnostics = SubspaceWeightingDiagnostics(
        requested="CVA",
        applied="CVA",
        covariance_rank=covariance_rank,
        covariance_rows=covariance_rows,
    )
    return (*decomposition, square_root, diagnostics)
