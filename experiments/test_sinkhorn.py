import argparse
from dataclasses import dataclass

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.xpred_var_decoder import sinkhorn_loss, sinkhorn_divergence


@dataclass
class TrainCfg:
    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-3
    hidden_dim: int = 512
    depth: int = 5
    noise_dim: int = 128
    epsilon: float = 0.01
    n_iters: int = 50
    log_every: int = 100
    seed: int = 123
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        return torch.sigmoid(x)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(batch_size: int) -> DataLoader:
    tfm = transforms.ToTensor()
    ds = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)


def main(cfg: TrainCfg):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    loader = build_loader(cfg.batch_size)
    out_dim = 3 * 8 * 8
    model = MLP(cfg.noise_dim, cfg.hidden_dim, out_dim, cfg.depth).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    out_dir = "experiments/output"
    os.makedirs(out_dir, exist_ok=True)

    step = 0
    for ep in range(cfg.epochs):
        model.train()
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            target = F.interpolate(imgs, size=(8, 8), mode="area")
            target_flat = target.view(target.shape[0], -1)

            z = torch.randn(target.shape[0], cfg.noise_dim, device=device)
            pred_flat = model(z)

            loss = sinkhorn_loss(pred_flat, target_flat, epsilon=cfg.epsilon, n_iters=cfg.n_iters)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            if step % cfg.log_every == 0:
                with torch.no_grad():
                    targ_v_targ_loss = sinkhorn_loss(target_flat, target_flat, epsilon=cfg.epsilon, n_iters=cfg.n_iters)
                print(f"ep {ep+1} step {step} loss {loss.item():.4f} real_vs_real_loss {targ_v_targ_loss.item():.4f}")
            step += 1
        model.eval()
        with torch.no_grad():
            z = torch.randn(4, cfg.noise_dim, device=device)
            pred_flat = model(z)
            pred = pred_flat.view(4, 3, 8, 8)
            grid = make_grid(pred, nrow=4, padding=2)
            save_image(grid, f"{out_dir}/epoch_{ep+1:04d}_eps{cfg.epsilon}_niter{cfg.n_iters}_div.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Sinkhorn training on CIFAR-10 8x8 downsampled images.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--noise-dim", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--n-iters", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = TrainCfg(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        noise_dim=args.noise_dim,
        epsilon=args.epsilon,
        n_iters=args.n_iters,
        log_every=args.log_every,
        seed=args.seed,
    )
    main(cfg)
