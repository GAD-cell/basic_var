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
from transformers import GPT2Config, GPT2Model
from eval.fid import fid_from_features, precision_recall_knn, precision_recall_knn_blockwise
from eval.features import get_dinov2_model, extract_dinov2_features


@dataclass
class XPredConfig:
    scales: Tuple[int, ...]          # increasing order, each divisible by patch_size
    patch_size: int                  # non-overlapping
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    dropout: float = 0.0
    decoder_type: str = "gpt2"       # "gpt2" or "var"
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0
    attn_l2_norm: bool = False
    shared_aln: bool = False
    num_classes: int = 1000
    cond_drop_prob: float = 0.1
    use_noise_seed: bool = True
    noise_dim: int = 256
    noise_scale: float = 1.0
    input_noise_base: float = 0.0
    input_noise_decay: float = 0.7


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

def _check_scales(scales: Tuple[int, ...], p: int):
    assert len(scales) >= 2, "Need at least two scales."
    assert all(scales[i] < scales[i + 1] for i in range(len(scales) - 1)), "scales must be increasing."
    assert all(s % p == 0 for s in scales), "each scale must be divisible by patch_size."


def patchify(img_bchw: torch.Tensor, p: int) -> torch.Tensor:
    """
    Non-overlapping patchify.
    img_bchw: [B, 3, H, W]
    returns: [B, n_patches, p*p*3]
    """
    B, C, H, W = img_bchw.shape
    assert H % p == 0 and W % p == 0, "H and W must be divisible by patch_size."
    patches = img_bchw.unfold(2, p, p).unfold(3, p, p)  # B, C, Hp, Wp, p, p
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # B, Hp, Wp, C, p, p
    patches = patches.view(B, -1, C * p * p)
    return patches


def unpatchify(patches_bnp: torch.Tensor, p: int, H: int, W: int) -> torch.Tensor:
    """
    patches_bnp: [B, n_patches, p*p*3]
    returns: [B, 3, H, W]
    """
    B, N, D = patches_bnp.shape
    C = 3
    hp = H // p
    wp = W // p
    assert N == hp * wp, "n_patches mismatch."
    patches = patches_bnp.view(B, hp, wp, C, p, p)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    return patches.view(B, C, H, W)


def build_blockwise_mask(block_sizes: List[int], device: torch.device) -> torch.Tensor:
    """
    Returns additive attention mask of shape [1, 1, L, L] with 0 for allowed and -inf for disallowed.
    Block-wise causal: tokens in scale k can attend to scales <= k.
    """
    L = sum(block_sizes)
    scale_ids = []
    for k, n in enumerate(block_sizes):
        scale_ids.extend([k] * n)
    scale_ids = torch.tensor(scale_ids, device=device)
    # allow if key_scale <= query_scale
    q = scale_ids.view(L, 1)
    k = scale_ids.view(1, L)
    allow = (k <= q)
    mask = torch.zeros((L, L), device=device)
    mask = mask.masked_fill(~allow, float("-inf"))
    return mask.view(1, 1, L, L)


class XPredNextScale(nn.Module):
    def __init__(self, cfg: XPredConfig):
        super().__init__()
        _check_scales(cfg.scales, cfg.patch_size)
        self.cfg = cfg
        self.scales = cfg.scales
        self.p = cfg.patch_size
        self.num_scales = len(cfg.scales)

        # tokens per scale
        self.block_sizes = [(s // self.p) ** 2 for s in self.scales]
        self.L = sum(self.block_sizes)
        self.patch_dim = self.p * self.p * 3

        # embeddings
        self.patch_embed = nn.Linear(self.patch_dim, cfg.d_model)
        self.pos_1L = self._build_sinusoidal_pos().unsqueeze(0)  # [1, L, d_model]
        self.scale_embed = nn.Embedding(self.num_scales, cfg.d_model)
        self.class_embed = nn.Embedding(cfg.num_classes + 1, cfg.d_model)  # last = uncond

        self.start_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.noise_proj = nn.Linear(cfg.noise_dim, cfg.d_model) if cfg.use_noise_seed else None

        if cfg.decoder_type == "gpt2":
            print("Using GPT-2 decoder.")
            gpt_cfg = GPT2Config(
                n_embd=cfg.d_model,
                n_layer=cfg.n_layer,
                n_head=cfg.n_head,
                n_positions=self.L,
                resid_pdrop=cfg.dropout,
                embd_pdrop=cfg.dropout,
                attn_pdrop=cfg.dropout,
            )
            self.decoder = GPT2Model(gpt_cfg)
            # disable token-level causal mask so we can use block-wise causal mask
            for block in self.decoder.h:
                block.attn.is_causal = False
        elif cfg.decoder_type == "var":
            print("Using VAR decoder.")
            from models.xpred_var_decoder import XPredVARDecoder
            self.decoder = XPredVARDecoder(
                d_model=cfg.d_model,
                depth=cfg.n_layer,
                num_heads=cfg.n_head,
                mlp_ratio=cfg.mlp_ratio,
                drop=cfg.dropout,
                attn_drop=cfg.dropout,
                drop_path_rate=cfg.drop_path_rate,
                norm_eps=1e-6,
                shared_aln=cfg.shared_aln,
                attn_l2_norm=cfg.attn_l2_norm,
                flash_if_available=False,
                fused_if_available=True,
            )
        else:
            raise ValueError(f"unknown decoder_type: {cfg.decoder_type}")

        self.head = nn.Linear(cfg.d_model, self.patch_dim)

    def _encode_condition(self, labels: Optional[torch.Tensor], B: int, device: torch.device) -> torch.Tensor:
        if labels is None:
            labels = torch.full((B,), self.cfg.num_classes, device=device, dtype=torch.long)
        return self.class_embed(labels)

    def _build_sinusoidal_pos(self) -> torch.Tensor:
        """
        Build within-scale 2D sinusoidal positional encoding.
        Each patch position is mapped to original image coordinates (largest scale)
        so that top-left patches differ across scales.
        Returns [L, d_model].
        """
        d_model = self.cfg.d_model
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D sinusoidal encoding.")
        sK = self.scales[-1]
        pos_list = []
        for s in self.scales:
            g = s // self.p
            scale_factor = sK / s
            # patch centers in original-image coordinate system
            ys = (torch.arange(g, dtype=torch.float32) + 0.5) * self.p * scale_factor
            xs = (torch.arange(g, dtype=torch.float32) + 0.5) * self.p * scale_factor
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            # normalize to [0, 1]
            yy = yy / sK
            xx = xx / sK

            pos = self._sinusoidal_2d(xx.reshape(-1), yy.reshape(-1), d_model)
            pos_list.append(pos)
        return torch.cat(pos_list, dim=0)

    @staticmethod
    def _sinusoidal_2d(x: torch.Tensor, y: torch.Tensor, d_model: int) -> torch.Tensor:
        """
        x, y: [N] in [0,1]
        returns [N, d_model]
        """
        half = d_model // 2
        quarter = half // 2
        omega = torch.arange(quarter, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / quarter))

        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        out_x = torch.cat([torch.sin(x * omega), torch.cos(x * omega)], dim=1)
        out_y = torch.cat([torch.sin(y * omega), torch.cos(y * omega)], dim=1)
        return torch.cat([out_x, out_y], dim=1)

    def _build_inputs(self, imgs_bchw: torch.Tensor, labels: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          inputs_embeds: [B, L, d_model]
          targets: [B, L, patch_dim]
        """
        B, _, _, _ = imgs_bchw.shape
        device = imgs_bchw.device

        # downsample to each scale
        imgs_k = [F.adaptive_avg_pool2d(imgs_bchw, (s, s)) for s in self.scales]
        targets = [patchify(im, self.p) for im in imgs_k]  # P_k

        def add_scale_noise(x: torch.Tensor, k: int) -> torch.Tensor:
            if self.cfg.input_noise_base <= 0:
                return x
            sigma = self.cfg.input_noise_base * (self.cfg.input_noise_decay ** k)
            return x + torch.randn_like(x) * sigma

        # build input blocks
        blocks = []
        # scale 1: start token (learned), optionally add noise + cond
        n1 = self.block_sizes[0]
        start = self.start_token.expand(B, n1, -1)
        if self.cfg.use_noise_seed:
            noise = torch.randn(B, self.cfg.noise_dim, device=device)
            start = start + self.noise_proj(noise).unsqueeze(1) * self.cfg.noise_scale
        start = add_scale_noise(start, 0)
        blocks.append(start)

        # scales 2..K: upsample previous scale to current size, then patchify
        for k in range(1, self.num_scales):
            prev = imgs_k[k - 1]
            up = F.interpolate(prev, size=(self.scales[k], self.scales[k]), mode="bicubic", align_corners=False)
            patches = patchify(up, self.p)
            block = self.patch_embed(patches)
            block = add_scale_noise(block, k)
            blocks.append(block)

        inputs = torch.cat(blocks, dim=1)  # [B, L, d_model]

        # add positional + scale embeddings
        pos = self.pos_1L.to(device=device)
        scale_ids = []
        for k, n in enumerate(self.block_sizes):
            scale_ids.extend([k] * n)
        scale_ids = torch.tensor(scale_ids, device=device)
        scale = self.scale_embed(scale_ids).unsqueeze(0)
        cond = self._encode_condition(labels, B, device).unsqueeze(1)
        inputs = inputs + pos + scale + cond

        targets = torch.cat(targets, dim=1)  # [B, L, patch_dim]
        return inputs, targets

    def forward(self, imgs_bchw: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns MSE loss.
        """
        B, _, _, _ = imgs_bchw.shape
        device = imgs_bchw.device

        # conditional dropout (CFG training)
        if labels is not None and self.cfg.cond_drop_prob > 0:
            drop = torch.rand(B, device=device) < self.cfg.cond_drop_prob
            labels = labels.clone()
            labels[drop] = self.cfg.num_classes  # uncond

        inputs, targets = self._build_inputs(imgs_bchw, labels)
        attn_mask = build_blockwise_mask(self.block_sizes, device=device)
        if self.cfg.decoder_type == "gpt2":
            out = self.decoder(inputs_embeds=inputs, attention_mask=attn_mask)
            h = out.last_hidden_state
        else:
            cond_vec = self._encode_condition(labels, B, device)
            h = self.decoder(inputs, cond_vec, attn_mask)
            h = self.decoder.apply_head_norm(h, cond_vec)
        preds = self.head(h)
        return F.mse_loss(preds, targets)

    @torch.no_grad()
    def generate(
        self,
        B: int,
        labels: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.5,
    ) -> torch.Tensor:
        """
        Returns generated image at the largest scale, shape [B, 3, sK, sK].
        Uses KV caching; no blockwise mask needed at inference.
        """
        device = next(self.parameters()).device
        if labels is None:
            labels = torch.randint(0, self.cfg.num_classes, (B,), device=device)
        uncond = torch.full((B,), self.cfg.num_classes, device=device, dtype=torch.long)

        # scale 1 input (start token)
        n1 = self.block_sizes[0]
        start = self.start_token.expand(B, n1, -1)
        if self.cfg.use_noise_seed:
            noise = torch.randn(B, self.cfg.noise_dim, device=device)
            start = start + self.noise_proj(noise).unsqueeze(1) * self.cfg.noise_scale
        if self.cfg.input_noise_base > 0:
            sigma0 = self.cfg.input_noise_base
            start = start + torch.randn_like(start) * sigma0

        # add pos/scale/cond
        pos = self.pos_1L.to(device=device)
        scale_ids = []
        for k, n in enumerate(self.block_sizes):
            scale_ids.extend([k] * n)
        scale_ids = torch.tensor(scale_ids, device=device)
        scale = self.scale_embed(scale_ids).unsqueeze(0)

        def add_cond(x, y):
            return x + pos[:, : x.shape[1]] + scale[:, : x.shape[1]] + self.class_embed(y).unsqueeze(1)

        # first step: predict scale 1
        if self.cfg.decoder_type == "var":
            self.decoder.enable_kv_cache(True)

        inp_cond = add_cond(start, labels)
        inp_uncond = add_cond(start, uncond)

        if self.cfg.decoder_type == "gpt2":
            past = None
            out_cond = self.decoder(inputs_embeds=inp_cond, use_cache=True, past_key_values=past)
            out_uncond = self.decoder(inputs_embeds=inp_uncond, use_cache=True, past_key_values=past)
            h_cond = out_cond.last_hidden_state
            h_uncond = out_uncond.last_hidden_state
            past = out_cond.past_key_values
        else:
            cond_vec = self._encode_condition(labels, B, device)
            uncond_vec = self._encode_condition(uncond, B, device)
            h_cond = self.decoder(inp_cond, cond_vec, attn_bias=None)
            h_uncond = self.decoder(inp_uncond, uncond_vec, attn_bias=None)
            h_cond = self.decoder.apply_head_norm(h_cond, cond_vec)
            h_uncond = self.decoder.apply_head_norm(h_uncond, uncond_vec)

        pred = (1 + cfg_scale) * self.head(h_cond) - cfg_scale * self.head(h_uncond)

        # reconstruct scale-1 image
        s1 = self.scales[0]
        img_k = unpatchify(pred, self.p, s1, s1)

        # subsequent scales
        for k in range(1, self.num_scales):
            sk = self.scales[k]
            up = F.interpolate(img_k, size=(sk, sk), mode="bicubic", align_corners=False)
            patches = patchify(up, self.p)
            inp = self.patch_embed(patches)
            if self.cfg.input_noise_base > 0:
                sigma = self.cfg.input_noise_base * (self.cfg.input_noise_decay ** k)
                inp = inp + torch.randn_like(inp) * sigma
            inp_cond = add_cond(inp, labels)
            inp_uncond = add_cond(inp, uncond)

            if self.cfg.decoder_type == "gpt2":
                out_cond = self.decoder(inputs_embeds=inp_cond, use_cache=True, past_key_values=past)
                out_uncond = self.decoder(inputs_embeds=inp_uncond, use_cache=True, past_key_values=past)
                h_cond = out_cond.last_hidden_state
                h_uncond = out_uncond.last_hidden_state
                past = out_cond.past_key_values
            else:
                cond_vec = self._encode_condition(labels, B, device)
                uncond_vec = self._encode_condition(uncond, B, device)
                h_cond = self.decoder(inp_cond, cond_vec, attn_bias=None)
                h_uncond = self.decoder(inp_uncond, uncond_vec, attn_bias=None)
                h_cond = self.decoder.apply_head_norm(h_cond, cond_vec)
                h_uncond = self.decoder.apply_head_norm(h_uncond, uncond_vec)
            pred = (1 + cfg_scale) * self.head(h_cond) - cfg_scale * self.head(h_uncond)

            img_k = unpatchify(pred, self.p, sk, sk)

        if self.cfg.decoder_type == "var":
            self.decoder.enable_kv_cache(False)
        return img_k


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
        imgs = model.generate(B=cur).clamp(0, 1)
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
    gen = model.generate(B=n_samples).clamp(0, 1)
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
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_name = f"{model.cfg.decoder_type}-d{model.cfg.d_model}-L{model.cfg.n_layer}"
            train_cfg.wandb_run_name = f"{ts}-dataset-{model_name}-ep{train_cfg.epochs}"
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
                with torch.amp.autocast():
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
    _train_loop(model, train_cfg, ld, train_cfg.real_features_path, train_ds=train_ds)


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

    _train_loop(model, train_cfg, train_ds, str(features_path), progress_bar=progress_bar, train_ds=train_ds)

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
        batch_size=32, 
        eval_every_n_epochs=5, 
        n_eval_samples=5000, 
        real_features_path="data/cifar10_train_dinov2_features.pt", 
        real_subset=50000, knn_k=3, use_amp=_use_amp(device), 
        device=device.type, 
        use_wandb=True,
        wandb_run_name="cifar10-var-L4-d128-e1000"
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
        use_noise_seed=False,
        input_noise_base=0.05,
        input_noise_decay=0.6,
        num_classes=10,
    )

    model = XPredNextScale(cfg)
    train_cifar10(model, train_cfg, data_root="data", feature_split="train", feature_batch_size=128, force_recompute_features=False)
