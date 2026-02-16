import torch
from typing import Tuple

def _sqrtm_psd(mat: torch.Tensor) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=0)
    return (eigvecs * eigvals.sqrt().unsqueeze(0)) @ eigvecs.T


def fid_from_features(real: torch.Tensor, fake: torch.Tensor) -> float:
    mu_r, mu_f = real.mean(0), fake.mean(0)
    cov_r = torch.cov(real.T)
    cov_f = torch.cov(fake.T)
    cov_sqrt = _sqrtm_psd(cov_r @ cov_f)
    fid = (mu_r - mu_f).pow(2).sum() + torch.trace(cov_r + cov_f - 2 * cov_sqrt)
    return float(fid)

def precision_recall_knn(real: torch.Tensor, fake: torch.Tensor, k: int = 5) -> Tuple[float, float]:
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