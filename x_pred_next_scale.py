import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import GPT2Config, GPT2Model
from eval.fid import fid_from_features, precision_recall_knn
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


@dataclass
class TrainConfig:
    train_data_path: str = "/path/to/imagenet"
    epochs: int = 10
    batch_size: int = 32
    workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.05
    grad_accum: int = 1
    eval_every_n_epochs: int = 1
    n_eval_samples: int = 5000
    real_features_path: str = "/path/to/imagenet_dinov2_features.pt"
    real_subset: int = 50000
    knn_k: int = 5
    use_amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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

        # build input blocks
        blocks = []
        # scale 1: start token (learned), optionally add noise + cond
        n1 = self.block_sizes[0]
        start = self.start_token.expand(B, n1, -1)
        if self.cfg.use_noise_seed:
            noise = torch.randn(B, self.cfg.noise_dim, device=device)
            start = start + self.noise_proj(noise).unsqueeze(1) * self.cfg.noise_scale
        blocks.append(start)

        # scales 2..K: upsample previous scale to current size, then patchify
        for k in range(1, self.num_scales):
            prev = imgs_k[k - 1]
            up = F.interpolate(prev, size=(self.scales[k], self.scales[k]), mode="bilinear", align_corners=False)
            patches = patchify(up, self.p)
            blocks.append(self.patch_embed(patches))

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
            up = F.interpolate(img_k, size=(sk, sk), mode="bilinear", align_corners=False)
            patches = patchify(up, self.p)
            inp = self.patch_embed(patches)
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
    device = next(model.parameters()).device
    dinov2 = get_dinov2_model(device)

    fake_feats = []
    remaining = n_samples
    while remaining > 0:
        cur = min(batch_size, remaining)
        imgs = model.generate(B=cur).clamp(0, 1)
        feats = extract_dinov2_features(dinov2, imgs)
        fake_feats.append(feats.cpu())
        remaining -= cur
    fake_feats = torch.cat(fake_feats, dim=0)

    real_feats = torch.load(real_features_path, map_location="cpu")
    if real_subset > 0 and real_feats.shape[0] > real_subset:
        idx = torch.randperm(real_feats.shape[0])[:real_subset]
        real_feats = real_feats[idx]

    fid = fid_from_features(real_feats, fake_feats)
    precision, recall = precision_recall_knn(real_feats, fake_feats, k=knn_k)
    return {"fid": fid, "precision": precision, "recall": recall}


def train(model: XPredNextScale, train_cfg: TrainConfig):
    device = torch.device(train_cfg.device)
    model.to(device)

    sK = model.scales[-1]
    ld = build_dataloader(train_cfg.train_data_path, sK, train_cfg.batch_size, train_cfg.workers)

    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=train_cfg.epochs * len(ld))
    scaler = torch.cuda.amp.GradScaler() if train_cfg.use_amp and device.type == "cuda" else None

    for ep in range(train_cfg.epochs):
        model.train()
        optim.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        n_batches = 0
        for i, batch in enumerate(ld):
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = train_one_batch(model, batch, optim, scaler, train_cfg.grad_accum)
            else:
                loss = train_one_batch(model, batch, optim, scaler, train_cfg.grad_accum)
            # train_one_batch returns loss scaled by grad_accum
            epoch_loss += float(loss) * train_cfg.grad_accum
            n_batches += 1

            if (i + 1) % train_cfg.grad_accum == 0:
                if scaler is not None:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)
                scheduler.step()

        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
            print(f"[train ep {ep+1}] loss={avg_loss:.6f}")

        if (ep + 1) % train_cfg.eval_every_n_epochs == 0:
            model.eval()
            metrics = evaluate_model(
                model=model,
                n_samples=train_cfg.n_eval_samples,
                batch_size=min(train_cfg.batch_size, train_cfg.n_eval_samples),
                real_features_path=train_cfg.real_features_path,
                real_subset=train_cfg.real_subset,
                knn_k=train_cfg.knn_k,
            )
            print(f"[eval ep {ep+1}] {metrics}")

def test():
    cfg = XPredConfig(scales=(16, 32, 64), patch_size=16, d_model=256, n_layer=4, n_head=4, decoder_type="var")
    model = XPredNextScale(cfg)
    x = torch.randn(2, 3, 64, 64)
    y = torch.randint(0, cfg.num_classes, (2,))
    loss = model(x, y)
    print("loss:", float(loss))

if __name__ == "__main__":
    test()
