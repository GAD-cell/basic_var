import math
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    wandb = None
from eval.fid import fid_from_features, precision_recall_knn_blockwise
from eval.features import get_dinov2_model, extract_dinov2_features
from models.xpred_var_decoder import XPredNextScale, XPredConfig


@dataclass
class TrainConfig:
    train_data_path: str = "data/"
    epochs: int = 10
    batch_size: int = 32
    workers: int = 0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_accum: int = 1
    ckpt_every_n_steps: int = 0
    eval_every_n_epochs: int = 10
    n_eval_samples: int = 5000
    real_features_path: str = "data/"
    real_subset: int = 50000
    knn_k: int = 3
    use_amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_wandb: bool = False
    wandb_project: str = "x_pred_next_scale"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def _use_amp(device: torch.device) -> bool:
    if device.type == "cuda":
        return True
    return False


def build_dataloader(train_path: str, img_size: int, batch_size: int, workers: int) -> DataLoader:
    tfm = transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )
    ds = datasets.ImageFolder(train_path, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)


def build_imagefolder_dataset(train_path: str, img_size: int):
    tfm = transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )
    return datasets.ImageFolder(train_path, transform=tfm)


def train_one_batch(model: nn.Module, batch, optimizer, scaler, grad_accum: int):
    imgs, labels = batch
    device = next(model.parameters()).device
    imgs = imgs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    loss = model(imgs, labels)
    loss = loss / grad_accum
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    return loss.detach()



@torch.no_grad()
def evaluate_model(
    dataset: datasets.ImageFolder,
    model: XPredNextScale,
    n_samples: int,
    batch_size: int,
    real_features_path: str,
    real_subset: int,
    knn_k: int,
):
    print("Evaluating model...")
    device = next(model.parameters()).device
    dinov2 = get_dinov2_model(device)

    fake_feats = []
    remaining = n_samples
    print(f"Generating {n_samples} samples for evaluation...")
    while remaining > 0:
        cur = min(batch_size, remaining)

        # retrieve a batch of real lowest-scale images to condition on
        idx = torch.randint(0, len(dataset), (cur,))
        img = torch.stack([dataset[i][0] for i in idx], dim=0).to(device)
        low_sc = F.interpolate(img, size=(model.scales[0], model.scales[0]), mode="area")

        imgs = model.generate(B=cur, low_sc=low_sc).clamp(0, 1)
        feats = extract_dinov2_features(dinov2, imgs)
        fake_feats.append(feats.cpu())
        remaining -= cur
    fake_feats = torch.cat(fake_feats, dim=0)

    print(f"Loading real features from {real_features_path}...")
    real_feats = torch.load(real_features_path, map_location="cpu")
    if real_subset > 0 and real_feats.shape[0] > real_subset:
        idx = torch.randperm(real_feats.shape[0])[:real_subset]
        real_feats = real_feats[idx]

    print("Computing FID...")
    fid = fid_from_features(real_feats, fake_feats)
    print("Computing precision and recall...")
    precision, recall = precision_recall_knn_blockwise(real_feats, fake_feats, k=knn_k)
    return {"fid": fid, "precision": precision, "recall": recall}

def save_model(
    model: XPredNextScale,
    filename: str,
    train_cfg: TrainConfig,
    step: int,
    epoch: int,
    fid: Optional[float] = None,
):
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / filename
    torch.save(
        {
            "model": model.state_dict(),
            "step": step,
            "epoch": epoch,
            "fid": fid,
            "train_cfg": vars(train_cfg),
            "model_cfg": vars(model.cfg),
        },
        ckpt_path,
    )


@torch.no_grad()
def _save_mosaic(gen_imgs: torch.Tensor, nn_imgs: torch.Tensor, out_path: Path):
    """
    Save a 2-row mosaic: generated images on top, nearest neighbors below.
    """
    assert gen_imgs.shape == nn_imgs.shape
    b = gen_imgs.shape[0]
    stack = torch.cat([gen_imgs, nn_imgs], dim=0)
    grid = make_grid(stack, nrow=b, padding=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)


@torch.no_grad()
def _nearest_neighbors_mosaic(
    model: XPredNextScale,
    train_ds : datasets.ImageFolder,
    real_features_path: str,
    out_path: Path,
    device: torch.device,
    n_samples: int = 3,
):
    model.eval()

    # retrieve a batch of real lowest-scale images to condition on
    idx = torch.randint(0, len(train_ds), (n_samples,))
    img = torch.stack([train_ds[i][0] for i in idx], dim=0).to(device)
    low_sc = F.interpolate(img, size=(model.scales[0], model.scales[0]), mode="area")

    gen = model.generate(B=n_samples, low_sc=low_sc).clamp(0, 1)
    dinov2 = get_dinov2_model(device)
    gen_feats = extract_dinov2_features(dinov2, gen)

    real_feats = torch.load(real_features_path, map_location="cpu")
    if real_feats.dim() != 2:
        raise ValueError(f"expected real features [N, D], got {tuple(real_feats.shape)}")
    real_feats = real_feats.to(device)

    dist = torch.cdist(gen_feats, real_feats)
    nn_idx = dist.argmin(dim=1).tolist()
    nn_imgs = []
    for idx in nn_idx:
        img, _ = train_ds[idx]
        nn_imgs.append(img)
    nn_imgs = torch.stack(nn_imgs, dim=0).to(device)

    _save_mosaic(gen, nn_imgs, out_path)

def _train_loop(
    model: XPredNextScale,
    train_cfg: TrainConfig,
    train_ds: datasets.ImageFolder,
    real_features_path: str,
    progress_bar: bool = False,
):
    print("Starting training...")
    device = torch.device(train_cfg.device)
    model.to(device)

    ld = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.workers,
        pin_memory=True,
        drop_last=True,
    )

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_fid = float("inf")

    run = None
    if train_cfg.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed but use_wandb=True")
        if not train_cfg.wandb_run_name:
            model_name = f"{model.cfg.decoder_type}-d{model.cfg.d_model}-L{model.cfg.n_layer}-H{model.cfg.n_head}"
            train_cfg.wandb_run_name = f"dataset-{model_name}-ep{train_cfg.epochs}"
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_cfg.wandb_run_name = ts + "-" + train_cfg.wandb_run_name
        run = wandb.init(
            project=train_cfg.wandb_project,
            name=train_cfg.wandb_run_name,
            entity=train_cfg.wandb_entity,
            config={**vars(train_cfg), **vars(model.cfg)},
        )

    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=train_cfg.epochs * len(ld))
    scaler = torch.amp.GradScaler() if train_cfg.use_amp and device.type == "cuda" else None

    step = 0
    for ep in range(train_cfg.epochs):
        model.train()
        optim.zero_grad(set_to_none=True)
        if progress_bar:
            pbar = tqdm(ld, desc=f"Epoch {ep+1}", leave=False)
        else:
            pbar = ld
            print(f"Epoch {ep+1}/{train_cfg.epochs}")
        for i, batch in enumerate(pbar):
            if scaler is not None:
                with torch.amp.autocast(device_type=device.type):
                    loss = train_one_batch(model, batch, optim, scaler, train_cfg.grad_accum)
            else:
                loss = train_one_batch(model, batch, optim, scaler, train_cfg.grad_accum)
            # train_one_batch returns loss scaled by grad_accum
            grad_norm = None
            if (i + 1) % train_cfg.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optim)
                total_norm_sq = 0.0
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm_sq += param_norm.item() ** 2
                grad_norm = total_norm_sq ** 0.5
            if progress_bar:
                postfix = {"loss": f"{(loss*train_cfg.grad_accum).item():.6f}"}
                if grad_norm is not None:
                    postfix["grad_norm"] = f"{grad_norm:.4f}"
                pbar.set_postfix(postfix)

            if train_cfg.use_wandb and run is not None:
                log_payload = {"loss": (loss * train_cfg.grad_accum).item()}
                if grad_norm is not None:
                    log_payload["grad_norm"] = grad_norm
                wandb.log(log_payload, step=step)

            # Weight update
            if (i + 1) % train_cfg.grad_accum == 0:
                if scaler is not None:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)
                scheduler.step()

            if train_cfg.ckpt_every_n_steps and step % train_cfg.ckpt_every_n_steps == 0:
                save_model(model, f"step_{step}.pt", train_cfg, step, ep + 1)

            step += 1

        if (ep + 1) % train_cfg.eval_every_n_epochs == 0:
            model.eval()
            metrics = evaluate_model(
                dataset=train_ds,
                model=model,
                n_samples=train_cfg.n_eval_samples,
                batch_size=min(train_cfg.batch_size, train_cfg.n_eval_samples),
                real_features_path=real_features_path,
                real_subset=train_cfg.real_subset,
                knn_k=train_cfg.knn_k,
            )
            print(f"[eval ep {ep+1}] {metrics}")
            fid_val = float(metrics.get("fid", float("inf")))
            if fid_val < best_fid:
                best_fid = fid_val
                save_model(model, "best.pt", train_cfg, step, ep + 1, fid=best_fid)
            if train_cfg.use_wandb and run is not None:
                wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
            mosaic_path = Path("mosaics") / f"mosaic_step_{step}.png"
            _nearest_neighbors_mosaic(
                model=model,
                train_ds=train_ds,
                real_features_path=real_features_path,
                out_path=mosaic_path,
                device=device,
                n_samples=3,
            )

    if train_cfg.use_wandb and run is not None:
        run.finish()


def train(model: XPredNextScale, train_cfg: TrainConfig):
    sK = model.scales[-1]
    ld = build_dataloader(train_cfg.train_data_path, sK, train_cfg.batch_size, train_cfg.workers)
    train_ds = build_imagefolder_dataset(train_cfg.train_data_path, sK)
    _train_loop(model, train_cfg, ld, train_cfg.real_features_path)


def _preprocess_cifar10_features(
    root: Path,
    device: torch.device,
    split: str,
    batch_size: int,
    out_path: Path,
    progress_bar: bool = False,
):
    is_train = split == "train"
    ds = datasets.CIFAR10(root=str(root), train=is_train, download=True, transform=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model = get_dinov2_model(device)
    feats = []
    pbar = tqdm(dl, desc=f"dinov2 {split}", leave=False) if progress_bar else dl
    for imgs, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        cur = extract_dinov2_features(model, imgs)
        feats.append(cur.cpu())
    feats = torch.cat(feats, dim=0)
    torch.save(feats, out_path)


def train_cifar10(
    model: XPredNextScale,
    train_cfg: TrainConfig,
    data_root: str = "data",
    feature_split: str = "train",
    feature_batch_size: int = 128,
    force_recompute_features: bool = False,
    progress_bar: bool = False,
):
    device = torch.device(train_cfg.device)
    model.to(device)

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    sK = model.scales[-1]
    tfm = transforms.Compose(
        [
            transforms.Resize(sK, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(sK),
            transforms.ToTensor(),
        ]
    )

    print(f"Loading CIFAR-10 training data with image size {sK}x{sK}...")
    train_ds = datasets.CIFAR10(root=str(root), train=True, download=True, transform=tfm)

    feature_split = feature_split.lower()
    if feature_split not in {"train", "test"}:
        raise ValueError("feature_split must be 'train' or 'test'")
    features_path = root / f"cifar10_{feature_split}_dinov2_features.pt"
    if force_recompute_features or not features_path.exists():
        print(f"[cifar10] computing DINOv2 features ({feature_split}) -> {features_path}")
        _preprocess_cifar10_features(
            root=root,
            device=device,
            split=feature_split,
            batch_size=feature_batch_size,
            out_path=features_path,
            progress_bar=progress_bar,
        )
    else:
        print(f"[cifar10] using existing feature at {features_path}")

    _train_loop(model, train_cfg, train_ds, str(features_path), progress_bar=progress_bar)

def test():
    cfg = XPredConfig(scales=(16, 32, 64), patch_size=16, d_model=256, n_layer=4, n_head=4, decoder_type="var")
    model = XPredNextScale(cfg)
    x = torch.randn(2, 3, 64, 64)
    y = torch.randint(0, cfg.num_classes, (2,))
    loss = model(x, y)
    print("loss:", float(loss))

if __name__ == "__main__":
    device = pick_device()
    print(f"Using device: {device}")
    train_cfg = TrainConfig(
        epochs=1000, 
        batch_size=64, 
        eval_every_n_epochs=5, 
        n_eval_samples=5000, 
        real_features_path="data/cifar10_train_dinov2_features.pt", 
        real_subset=50000, knn_k=3, use_amp=_use_amp(device), 
        device=device.type, 
        use_wandb=True,
        wandb_run_name="cifar10-var-L4-H4-d128-e1000",
        ckpt_every_n_steps=40_000,
    )
    cfg = XPredConfig(
        scales=(4, 8, 16, 32),
        patch_size=4,
        d_model=128,
        n_layer=4,
        n_head=4,
        decoder_type="var",
        mlp_ratio=2.0,
        drop_path_rate=0.05,
        attn_l2_norm=True,
        shared_aln=False,
        cond_drop_prob=0.1,
        num_classes=10,
        first_scale_noise_std=0.1,
        loss="mse" # "mse" or "sink" or "mse_wo_s1"
    )

    model = XPredNextScale(cfg)
    train_cifar10(model, train_cfg, data_root="data", feature_split="train", feature_batch_size=128, force_recompute_features=False, progress_bar=False)
