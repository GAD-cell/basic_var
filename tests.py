import argparse
from pathlib import Path

import torch

from eval.fid import fid_from_features, precision_recall_knn


def sanity_check_metrics(
    features_path: str,
    subset: int = 20000,
    split_seed: int = 0,
    knn_k: int = 5,
):
    path = Path(features_path)
    if not path.exists():
        raise FileNotFoundError(f"features file not found: {features_path}")

    feats = torch.load(path, map_location="cpu")
    if feats.dim() != 2:
        raise ValueError(f"expected 2D features [N, D], got shape {tuple(feats.shape)}")

    n = feats.shape[0]
    if subset > 0 and subset < n:
        g = torch.Generator().manual_seed(split_seed)
        idx = torch.randperm(n, generator=g)[:subset]
        feats = feats[idx]
        n = feats.shape[0]

    if n < 2:
        raise ValueError("not enough samples to split")

    mid = n // 2
    real_a = feats[:mid]
    real_b = feats[mid:]

    fid = fid_from_features(real_a, real_b)
    precision, recall = precision_recall_knn(real_a, real_b, k=knn_k)
    return {"fid": fid, "precision": precision, "recall": recall, "n": n, "k": knn_k}


def main():
    parser = argparse.ArgumentParser(description="Sanity check: real vs real FID/PR")
    parser.add_argument("features", type=str, help="Path to features .pt file")
    parser.add_argument("--subset", type=int, default=20000, help="Random subset size (0 = all)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for split")
    parser.add_argument("--k", type=int, default=5, help="k for kNN precision/recall")
    args = parser.parse_args()

    metrics = sanity_check_metrics(
        features_path=args.features,
        subset=args.subset,
        split_seed=args.seed,
        knn_k=args.k,
    )
    print(metrics)


if __name__ == "__main__":
    main()
