import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import make_grid, save_image
from pathlib import Path

from models.xpred_var_decoder import XPredNextScale
from eval.features import get_dinov2_model, extract_dinov2_features

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
def _save_mosaic_all_sc(gen_imgs: torch.Tensor, nn_imgs: torch.Tensor, out_path: Path):
    """
    gen_imgs, nn_imgs: [B, S, 3, H, W]
    Produces a grid with 2*B rows and S columns:
      row 0: gen scales for sample 0
      row 1: nn  scales for sample 0
      row 2: gen scales for sample 1
      row 3: nn  scales for sample 1
      ...
    """
    assert gen_imgs.shape == nn_imgs.shape
    B, S, C, H, W = gen_imgs.shape

    rows = []
    for i in range(B):
        rows.append(gen_imgs[i])  # [S, C, H, W]
        rows.append(nn_imgs[i])   # [S, C, H, W]

    grid_batch = torch.cat(rows, dim=0)          # [2*B*S, C, H, W]
    grid = make_grid(grid_batch, nrow=S, padding=2)

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
    labels = torch.tensor([train_ds[i][1] for i in idx], device=device, dtype=torch.long)
    low_sc = F.interpolate(img, size=(model.scales[0], model.scales[0]), mode="area")

    gen = model.generate(B=n_samples, low_sc=low_sc, labels=labels).clamp(0, 1)
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

@torch.no_grad()
def _nearest_neighbors_mosaic_all_sc(
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
    labels = torch.tensor([train_ds[i][1] for i in idx], device=device, dtype=torch.long)
    low_sc = F.interpolate(img, size=(model.scales[0], model.scales[0]), mode="area")

    gen_all_sc = model.generate_all_scales(B=n_samples, low_sc=low_sc, labels=labels)
    gen_all_sc = [gen.clamp(0, 1) for gen in gen_all_sc]
    gen_sK = gen_all_sc[-1]
    dinov2 = get_dinov2_model(device)
    gen_feats = extract_dinov2_features(dinov2, gen_sK)

    real_feats = torch.load(real_features_path, map_location="cpu")
    if real_feats.dim() != 2:
        raise ValueError(f"expected real features [N, D], got {tuple(real_feats.shape)}")
    real_feats = real_feats.to(device)

    dist = torch.cdist(gen_feats, real_feats)
    nn_idx = dist.argmin(dim=1).tolist()

    sK = model.scales[-1]
    nn_imgs = []
    for idx in nn_idx:
        img, _ = train_ds[idx]
        img_all_sc = []
        for sc in model.scales:
            img_k = F.interpolate(img.unsqueeze(0), size=(sc, sc), mode="area").squeeze(0)
            img_k = F.interpolate(img_k.unsqueeze(0), size=(sK, sK), mode="bicubic").squeeze(0)
            img_all_sc.append(img_k)
        nn_imgs.append(torch.stack(img_all_sc, dim=0))  # [num_scales, 3, sK, sK]
    nn_imgs = torch.stack(nn_imgs, dim=0).to(device) # [n_samples, num_scales, 3, sK, sK]

    gen_imgs = []
    for sc_idx in range(len(model.scales)):
        gen_s = F.interpolate(gen_all_sc[sc_idx], size=(sK, sK), mode="bicubic").squeeze(0)
        gen_imgs.append(gen_s)
    gen_imgs = torch.stack(gen_imgs, dim=1) # [n_samples, num_scales, 3, sK, sK]
    _save_mosaic_all_sc(gen_imgs, nn_imgs, out_path)