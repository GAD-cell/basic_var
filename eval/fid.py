import torch
from typing import Tuple, Optional

def _sqrtm_psd_sym(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Force symmetry (removes numerical asymmetry)
    mat = (mat + mat.transpose(-1, -2)) * 0.5
    # Jitter for stability
    mat = mat + eps * torch.eye(mat.shape[-1], device=mat.device, dtype=mat.dtype)

    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=0)
    return (eigvecs * eigvals.sqrt().unsqueeze(0)) @ eigvecs.transpose(-1, -2)


def fid_from_features(real: torch.Tensor, fake: torch.Tensor) -> float:
    mu_r, mu_f = real.mean(0), fake.mean(0)
    cov_r = torch.cov(real.T)
    cov_f = torch.cov(fake.T)

    sqrt_cov_r = _sqrtm_psd_sym(cov_r)
    a = sqrt_cov_r @ cov_f @ sqrt_cov_r
    covmean = _sqrtm_psd_sym(a)

    fid = (mu_r - mu_f).pow(2).sum() + torch.trace(cov_r + cov_f - 2.0 * covmean)

    return float(fid)

def precision_recall_knn(real: torch.Tensor, fake: torch.Tensor, k: int = 5) -> Tuple[float, float]:
    # Limit number of data points for memory efficiency
    limit = 5000
    if real.shape[0] > limit:
        real = real[torch.randperm(real.shape[0])[:limit]]
    if fake.shape[0] > limit:
        fake = fake[torch.randperm(fake.shape[0])[:limit]]

    dist_rr = torch.cdist(real, real)
    dist_rr.fill_diagonal_(float("inf"))
    radii_real = dist_rr.kthvalue(k, dim=1).values

    dist_rf = torch.cdist(fake, real)
    min_dist, nn_idx = dist_rf.min(dim=1)
    precision = (min_dist <= radii_real[nn_idx]).float().mean().item()

    dist_ff = torch.cdist(fake, fake)
    dist_ff.fill_diagonal_(float("inf"))
    radii_fake = dist_ff.kthvalue(k, dim=1).values
    dist_fr = torch.cdist(real, fake)
    min_dist_r, nn_idx_r = dist_fr.min(dim=1)
    recall = (min_dist_r <= radii_fake[nn_idx_r]).float().mean().item()
    return precision, recall


@torch.no_grad()
def _kth_nn_radii_blockwise(
    X: torch.Tensor,
    k: int,
    *,
    q_block: int = 1024,
    r_block: int = 8192,
) -> torch.Tensor:
    """
    Compute per-point radius = distance to the k-th nearest neighbor within X,
    WITHOUT materializing the full NxN distance matrix.

    Returns: radii [N] (float tensor)
    """

    device = X.device

    N = X.shape[0]
    radii = torch.empty((N,), device=device, dtype=X.dtype)

    # Precompute squared norms for fast squared L2 distances:
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    x_norm2 = (X * X).sum(dim=1)  # [N]

    for qs in range(0, N, q_block):
        qe = min(qs + q_block, N)
        Q = X[qs:qe]  # [Bq, D]
        q_norm2 = x_norm2[qs:qe]  # [Bq]

        # We keep the smallest (k+1) distances because self-distance is 0 for the same point.
        best = torch.full((qe - qs, k + 1), float("inf"), device=device, dtype=X.dtype)

        for rs in range(0, N, r_block):
            re = min(rs + r_block, N)
            R = X[rs:re]  # [Br, D]
            r_norm2 = x_norm2[rs:re]  # [Br]

            # squared distances [Bq, Br]
            # Use matmul for speed/memory; no huge persistent allocations beyond this block.
            dist2 = (q_norm2[:, None] + r_norm2[None, :] - 2.0 * (Q @ R.t())).clamp_min_(0.0)

            # If this ref block overlaps the query block, mask the diagonal self-distances.
            # This ensures "nearest neighbors" exclude the point itself.
            if rs < qe and re > qs:
                # overlap range in global indices: [max(qs, rs), min(qe, re))
                os = max(qs, rs)
                oe = min(qe, re)
                # convert to local indices
                q_idx = torch.arange(os - qs, oe - qs, device=device)
                r_idx = torch.arange(os - rs, oe - rs, device=device)
                dist2[q_idx, r_idx] = float("inf")

            # Merge candidates: concatenate current best with this block, then keep k+1 smallest
            merged = torch.cat([best, dist2], dim=1)  # [Bq, (k+1)+Br]
            best = torch.topk(merged, k=k + 1, dim=1, largest=False).values

        # k-th NN distance (excluding self) is the k-th smallest among (k+1) with self masked => index k
        radii[qs:qe] = best[:, k].sqrt()

    return radii


@torch.no_grad()
def _nn_min_dist_and_index_blockwise(
    Q: torch.Tensor,
    R: torch.Tensor,
    *,
    q_block: int = 1024,
    r_block: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each query in Q, find nearest neighbor in R (L2), blockwise.

    Returns:
      min_dist [Nq] (float tensor on device)
      nn_idx   [Nq] (long tensor on device)  indices into R
    """
    assert Q.dim() == 2 and R.dim() == 2
    Nq, Dq = Q.shape
    Nr, Dr = R.shape
    assert Dq == Dr, "feature dims must match"
    device = Q.device

    Q_dev = Q.to(device, non_blocking=True) if Q.device != device else Q
    R_dev = R.to(device, non_blocking=True) if R.device != device else R

    q_norm2 = (Q_dev * Q_dev).sum(dim=1)  # [Nq]
    r_norm2 = (R_dev * R_dev).sum(dim=1)  # [Nr]

    min_dist2 = torch.full((Nq,), float("inf"), device=device, dtype=Q_dev.dtype)
    nn_idx = torch.zeros((Nq,), device=device, dtype=torch.long)

    for qs in range(0, Nq, q_block):
        qe = min(qs + q_block, Nq)
        q = Q_dev[qs:qe]
        qn2 = q_norm2[qs:qe]

        best2 = torch.full((qe - qs,), float("inf"), device=device, dtype=Q_dev.dtype)
        besti = torch.zeros((qe - qs,), device=device, dtype=torch.long)

        for rs in range(0, Nr, r_block):
            re = min(rs + r_block, Nr)
            r = R_dev[rs:re]
            rn2 = r_norm2[rs:re]

            dist2 = (qn2[:, None] + rn2[None, :] - 2.0 * (q @ r.t())).clamp_min_(0.0)  # [Bq, Br]
            cur2, curi = dist2.min(dim=1)  # [Bq], [Bq] local indices

            better = cur2 < best2
            best2 = torch.where(better, cur2, best2)
            besti = torch.where(better, curi + rs, besti)

        min_dist2[qs:qe] = best2
        nn_idx[qs:qe] = besti

    return min_dist2.sqrt(), nn_idx


@torch.no_grad()
def precision_recall_knn_blockwise(
    real: torch.Tensor,
    fake: torch.Tensor,
    k: int = 5,
    *,
    q_block: int = 1024,
    r_block: int = 4096,
) -> Tuple[float, float]:
    """
    Blockwise (memory-safe) version of precision_recall_knn.

    It computes the SAME metric definition as your original implementation, but avoids O(N^2) memory.
    Time is still roughly O(N^2) in the worst case, but it won't allocate giant distance matrices.

    Args:
      real: [Nr, D]
      fake: [Nf, D]
      k:    kth neighbor radius
      device: "cuda" / "cpu" / torch.device / None (auto: use cuda if available else cpu)
      q_block: queries per block
      r_block: references per block

    Returns:
      precision (float), recall (float)
    """
    assert real.dim() == 2 and fake.dim() == 2, "real/fake must be [N, D]"
    assert real.shape[1] == fake.shape[1], "feature dims must match"
    Nr = real.shape[0]
    Nf = fake.shape[0]
    assert 1 <= k < Nr and 1 <= k < Nf, "k must be < number of points in each set"

    # Compute radii on device (can be GPU for speed)
    radii_real = _kth_nn_radii_blockwise(real, k, q_block=q_block, r_block=r_block)  # [Nr]
    radii_fake = _kth_nn_radii_blockwise(fake, k, q_block=q_block, r_block=r_block)  # [Nf]

    # Precision: each fake vs nearest real, test within real radius
    min_dist_fr, nn_idx_fr = _nn_min_dist_and_index_blockwise(
        fake, real, q_block=q_block, r_block=r_block
    )
    precision = (min_dist_fr <= radii_real[nn_idx_fr]).float().mean().item()

    # Recall: each real vs nearest fake, test within fake radius
    min_dist_rf, nn_idx_rf = _nn_min_dist_and_index_blockwise(
        real, fake, q_block=q_block, r_block=r_block
    )
    recall = (min_dist_rf <= radii_fake[nn_idx_rf]).float().mean().item()

    return precision, recall
