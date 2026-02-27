from typing import Optional, Tuple, List
from dataclasses import dataclass
from transformers import GPT2Config, GPT2Model

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.conv import ResidualConv
from eval.features import get_dinov2_model, extract_dinov2_features, extract_dinov2_features_with_grad


class XPredVARDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-6,
        shared_aln: bool = False,
        attn_l2_norm: bool = False,
        flash_if_available: bool = False,
        fused_if_available: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.C = d_model
        self.D = d_model
        self.depth = depth
        self.shared_aln = shared_aln

        def norm_layer(dim, elementwise_affine=True):
            return nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=elementwise_affine)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.shared_ada_lin = (
            nn.Sequential(nn.SiLU(inplace=False), nn.Linear(self.D, 6 * self.C))
            if shared_aln
            else nn.Identity()
        )

        self.blocks = nn.ModuleList(
            [
                AdaLNSelfAttn(
                    cond_dim=self.D,
                    shared_aln=shared_aln,
                    block_idx=i,
                    embed_dim=self.C,
                    norm_layer=norm_layer,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                    last_drop_p=0 if i == 0 else dpr[i - 1],
                    attn_l2_norm=attn_l2_norm,
                    flash_if_available=flash_if_available,
                    fused_if_available=fused_if_available,
                )
                for i in range(depth)
            ]
        )

        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)

    def enable_kv_cache(self, enable: bool):
        for b in self.blocks:
            b.attn.kv_caching(enable)

    def forward(
        self,
        x_BLC: torch.Tensor,
        cond_BD: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.shared_aln:
            cond = self.shared_ada_lin(cond_BD).view(-1, 1, 6, self.C)
        else:
            cond = cond_BD
        for b in self.blocks:
            x_BLC = b(x=x_BLC, cond_BD=cond, attn_bias=attn_bias)
        return x_BLC

    def apply_head_norm(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor) -> torch.Tensor:
        return self.head_nm(x_BLC, cond_BD)
    

@dataclass
class XPredConfig:
    scales: Tuple[int, ...]          # increasing order, each divisible by patch_size
    patch_size: int                  # non-overlapping
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    dropout: float = 0.0
    decoder_type: str = "var"       # "gpt2" or "var"
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0
    attn_l2_norm: bool = True
    shared_aln: bool = False
    num_classes: int = 1000
    cond_drop_prob: float = 0.1
    cfg_scale: float = 0.7
    first_scale_noise_std: float = 0.0
    fs_full_noise: bool = False
    loss: str = "mse" # "mse" or "sink" or "mse_wo_s1"
    sink_lbda: float = 1.0


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def _check_scales(scales: Tuple[int, ...], p: int):
    # assert len(scales) >= 2, "Need at least two scales."
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

def sinkhorn_loss(
    y,
    x,
    epsilon=1.0,
    n_iters=50
):
    """
    Differentiable Sinkhorn loss between model(z) and x.

    Parameters
    ----------
    y : Tensor (n, d)
        Source samples (given by model(z) where z sim mathcal{N}(0,1))
    x : Tensor (m, d)
        Target samples
    model : nn.Module
        Transport map T_theta
    epsilon : float
        Entropic regularization
    n_iters : int
        Number of Sinkhorn iterations

    Returns
    -------
    loss : scalar Tensor
    """

    n, m = y.shape[0], x.shape[0]
    assert y.shape[1] == x.shape[1], "samples must live in same dimension"
    device = y.device

    # Uniform marginals
    a = torch.full((n,), 1.0 / n, device=device)
    b = torch.full((m,), 1.0 / m, device=device)

    if device.type == "mps":
        # Cost matrix C_ij = ||y_i - x_j||^2 (manual, MPS-friendly)
        # C = ||y||^2 + ||x||^2 - 2 y x^T
        y_norm2 = (y * y).sum(dim=1)          # (n,)
        x_norm2 = (x * x).sum(dim=1)        # (m,)
        C = y_norm2[:, None] + x_norm2[None, :] - 2.0 * (y @ x.T)             # (n, m)
        C = C.clamp_min(0.0)  # guard against tiny negative values from roundoff
    else :
        C = torch.cdist(y, x, p=2) ** 2  # (n, m)
    C = C / y.shape[1]

    # Log-domain Sinkhorn variables
    log_a = torch.log(a)
    log_b = torch.log(b)

    with torch.no_grad():
        u = torch.zeros_like(log_a)
        v = torch.zeros_like(log_b)

        logK = -C / epsilon

        # Sinkhorn iterations
        for _ in range(n_iters):
            u = log_a - torch.logsumexp(logK + v[None,:], dim=1)
            v = log_b - torch.logsumexp(logK + u[:,None], dim=0)

        # Transport plan π
        pi = torch.exp(logK + u[:, None] + v[None, :])  # (n, m)

    # Sinkhorn cost
    loss = torch.sum(pi * C)

    return loss

def loss_sinkorn_s1_mse_s2sK(preds:torch.Tensor, targets:torch.Tensor, block_sizes:List[int], lbda:float = 1.0, use_dino_features:bool = True):
    """
    Applies sinkorn loss for first scale block, and MSE for later ones
    """

    # Prediction and target are (B, L, patch_dim)
    block_1_pred = preds[:, :block_sizes[0], :]
    later_blocks_pred = preds[:, block_sizes[0]:, :]
    block_1_target = targets[:, :block_sizes[0], :]
    later_blocks_target = targets[:, block_sizes[0]: , :]

    if use_dino_features:
        device = preds.device
        dino_model = get_dinov2_model(device)
        p = 4
        s1 = 32
        pred_img = unpatchify(block_1_pred, p, s1, s1)
        target_img = unpatchify(block_1_target, p, s1, s1)
        pred_feats = extract_dinov2_features_with_grad(
            dino_model,
            pred_img,
            freeze_model=True,
        )
        with torch.no_grad():
            target_feats = extract_dinov2_features(dino_model, target_img)
        sink_loss = sinkhorn_loss(pred_feats, target_feats)
    else:
        block1_pred_flat = block_1_pred.reshape(block_1_pred.shape[0], -1)
        block1_target_flat = block_1_target.reshape(block_1_target.shape[0], -1)
        sink_loss = sinkhorn_loss(block1_pred_flat, block1_target_flat)

    mse_loss = torch.nan_to_num(F.mse_loss(later_blocks_pred, later_blocks_target), nan=0.0)
    return mse_loss + lbda*sink_loss

def loss_mse_s2sK(preds:torch.Tensor, targets:torch.Tensor, block_sizes:List[int]):
    block_s2sK_pred = preds[:, block_sizes[0]:, :]
    blocks_s2sK_target = targets[:, block_sizes[0]: , :]
    return torch.nan_to_num(F.mse_loss(block_s2sK_pred, blocks_s2sK_target), nan=0.0)





class XPredNextScale(nn.Module):
    def __init__(self, cfg: XPredConfig):
        super().__init__()
        _check_scales(cfg.scales, cfg.patch_size)
        self.cfg = cfg
        self.scales = cfg.scales
        self.p = cfg.patch_size
        self.num_scales = len(cfg.scales)

        # tokens per scale (patches only)
        self.patch_block_sizes = [(s // self.p) ** 2 for s in self.scales]
        # add a start token to scale-0 block
        self.block_sizes = [self.patch_block_sizes[0] + 1] + self.patch_block_sizes[1:]
        self.L = sum(self.block_sizes)
        self.patch_L = sum(self.patch_block_sizes)
        self.patch_dim = self.p * self.p * 3

        # embeddings
        self.patch_embed = nn.Linear(self.patch_dim, cfg.d_model)
        pos_patches = self._build_sinusoidal_pos()  # [patch_L, d_model]
        start_pos = torch.zeros(1, cfg.d_model)
        self.pos_1L = torch.cat([start_pos, pos_patches], dim=0).unsqueeze(0)  # [1, L, d_model]
        self.scale_embed = nn.Embedding(self.num_scales, cfg.d_model)
        self.class_embed = nn.Embedding(cfg.num_classes + 1, cfg.d_model)  # last = uncond

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

        self.apply_conv: bool = False
        # Add conv modules after "unpatchify" for smoother images
        self.post_unpatch_convs = nn.ModuleList()
        for k in range(self.num_scales):
            sk = self.scales[k]
            convs = nn.Sequential(
                ResidualConv(dim=3, kernel_size=3), # dim is number of channels
            )
            self.post_unpatch_convs.append(convs)

    def _encode_condition(self, labels: Optional[torch.Tensor], B: int, device: torch.device) -> torch.Tensor:
        if labels is None:
            labels = torch.full((B,), self.cfg.num_classes, device=device, dtype=torch.long)
        return self.class_embed(labels)
    
    def _loss(self, preds:torch.Tensor, targets:torch.Tensor):
        if self.cfg.loss == "mse":
            return F.mse_loss(preds, targets)
        elif self.cfg.loss == "sink":
            return loss_sinkorn_s1_mse_s2sK(preds, targets, self.patch_block_sizes, self.cfg.sink_lbda)
        elif self.cfg.loss == "mse_wo_s1":
            return loss_mse_s2sK(preds, targets, self.patch_block_sizes)
        else:
            raise ValueError("loss must be either mse or sink or mse_wo_s1")
        
    def _build_first_scale(self, low_sc: torch.Tensor) -> torch.Tensor:
        """
        low_sc: [B, 3, s1, s1]
        returns: [B, 3, s1, s1]
        """
        low = low_sc
        if self.cfg.fs_full_noise:
            return torch.randn_like(low)
        if self.cfg.first_scale_noise_std > 0:
            low = low + torch.randn_like(low) * self.cfg.first_scale_noise_std
        return low

    def _build_sinusoidal_pos(self) -> torch.Tensor:
        """
        Build within-scale 2D sinusoidal positional encoding.
        Each patch position is mapped to original image coordinates (largest scale)
        so that top-left patches differ across scales.
        Returns [patch_L, d_model].
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
          targets: [B, patch_L, patch_dim]
        """
        B, _, _, _ = imgs_bchw.shape
        device = imgs_bchw.device

        # downsample to each scale
        imgs_k = [F.adaptive_avg_pool2d(imgs_bchw, (s, s)) for s in self.scales]
        targets = [patchify(im, self.p) for im in imgs_k]  # P_k

        # build input blocks
        blocks = []
        # first scale: start token + noisy lowest-scale image
        start = self._encode_condition(labels, B, device).unsqueeze(1)

        low = imgs_k[0]
        low = self._build_first_scale(low)
        low_patches = patchify(low, self.p)
        low_block = self.patch_embed(low_patches)
        block0 = torch.cat([start, low_block], dim=1)
        blocks.append(block0)

        # scales 2..K: upsample previous scale to current size, then patchify
        for k in range(1, self.num_scales):
            prev = imgs_k[k - 1]
            up = F.interpolate(prev, size=(self.scales[k], self.scales[k]), mode="bicubic", align_corners=False)
            patches = patchify(up, self.p)
            block = self.patch_embed(patches)
            blocks.append(block)

        inputs = torch.cat(blocks, dim=1)  # [B, L, d_model]

        # add positional + scale embeddings
        pos = self.pos_1L.to(device=device)
        scale_ids = []
        for k, n in enumerate(self.block_sizes):
            scale_ids.extend([k] * n)
        scale_ids = torch.tensor(scale_ids, device=device)
        scale = self.scale_embed(scale_ids).unsqueeze(0)
        inputs = inputs + pos + scale

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
        preds = preds[:, 1:, :]  # drop start token

        if self.apply_conv:
            # apply convs after unpatchify for smoother images before computing loss
            recon = []
            offset = 0
            for k in range(self.num_scales):
                sk = self.scales[k]
                patch_num = self.patch_block_sizes[k]
                pred_k = preds[:, offset:offset + patch_num, :]
                img_k = unpatchify(pred_k, self.p, sk, sk)
                img_k = self.post_unpatch_convs[k](img_k)
                recon_k = patchify(img_k, self.p)
                recon.append(recon_k)
                offset += patch_num
            recon = torch.cat(recon, dim=1)
            loss = self._loss(recon, targets)
            return loss
        
        return self._loss(preds, targets)

    @torch.no_grad()
    def generate(
        self,
        B: int,
        low_sc: torch.Tensor,  # [B, 3, s1, s1], clean lowest-scale image
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns generated image at the largest scale, shape [B, 3, sK, sK].
        Uses KV caching; no blockwise mask needed at inference.
        """
        device = next(self.parameters()).device
        if labels is None:
            labels = torch.randint(0, self.cfg.num_classes, (B,), device=device)
        uncond = torch.full((B,), self.cfg.num_classes, device=device, dtype=torch.long)

        # build pos/scale once
        pos = self.pos_1L.to(device=device)
        scale_ids = []
        for k, n in enumerate(self.block_sizes):
            scale_ids.extend([k] * n)
        scale_ids = torch.tensor(scale_ids, device=device)
        scale = self.scale_embed(scale_ids).unsqueeze(0)

        def add_pos(x, offset):
            # If you still want scale embedding, keep "+ scale[...]"
            return x + pos[:, offset:offset + x.shape[1]] + scale[:, offset:offset + x.shape[1]]
            # If you want ONLY positional embedding, drop the "+ scale[...]"

        # first step: predict scale 1
        offset = 0
        n1_block = self.block_sizes[0]

        if self.cfg.decoder_type == "var":
            # Keep prefix context via KV cache across scale blocks
            self.decoder.enable_kv_cache(True)

        # class conditioning only for first block
        start_cond = self._encode_condition(labels, B, device).unsqueeze(1)
        start_uncond = self._encode_condition(uncond, B, device).unsqueeze(1)


        # noisy lowest-scale image token (shared between cond/uncond)
        s1 = self.scales[0]
        #low = torch.zeros(B, 3, s1, s1, device=device)
        low = low_sc.to(device=device)
        low = self._build_first_scale(low)
        low_patches = patchify(low, self.p)
        low_block = self.patch_embed(low_patches)

        block_cond = torch.cat([start_cond, low_block], dim=1)
        block_uncond = torch.cat([start_uncond, low_block], dim=1)

        inp_cond = add_pos(block_cond, offset)
        inp_uncond = add_pos(block_uncond, offset)

        if self.cfg.decoder_type == "gpt2":
            past = None
            inp_b = torch.cat([inp_cond, inp_uncond], dim=0)
            out = self.decoder(inputs_embeds=inp_b, use_cache=True, past_key_values=past)
            h = out.last_hidden_state
            h_cond, h_uncond = h[:B], h[B:]
            past = out.past_key_values
        else:
            cond_vec = self._encode_condition(labels, B, device)
            uncond_vec = self._encode_condition(uncond, B, device)
            # batch-double: single forward for cond/uncond to keep KV cache aligned
            inp = torch.cat([inp_cond, inp_uncond], dim=0)
            cond = torch.cat([cond_vec, uncond_vec], dim=0)
            h = self.decoder(inp, cond, attn_bias=None)
            h = self.decoder.apply_head_norm(h, cond)
            h_cond, h_uncond = h[:B], h[B:]

        pred = (1 + self.cfg.cfg_scale) * self.head(h_cond) - self.cfg.cfg_scale * self.head(h_uncond)
        pred = pred[:, 1:, :]

        # reconstruct scale-1 image
        img_k = unpatchify(pred, self.p, s1, s1)
        if self.apply_conv:
            img_k = self.post_unpatch_convs[0](img_k)

        # subsequent scales
        offset += n1_block
        for k in range(1, self.num_scales):
            sk = self.scales[k]
            up = F.interpolate(img_k, size=(sk, sk), mode="bicubic", align_corners=False)
            patches = patchify(up, self.p)
            inp = self.patch_embed(patches)

            inp = add_pos(inp, offset)

            if self.cfg.decoder_type == "gpt2":
                # adding class embedding at each scale to match CFG design
                inp_cond = inp + self.class_embed(labels).unsqueeze(1)
                inp_uncond = inp + self.class_embed(uncond).unsqueeze(1)
                inp_b = torch.cat([inp_cond, inp_uncond], dim=0)
                out = self.decoder(inputs_embeds=inp_b, use_cache=True, past_key_values=past)
                h = out.last_hidden_state
                h_cond, h_uncond = h[:B], h[B:]
                past = out.past_key_values
            else:
                cond_vec = self._encode_condition(labels, B, device)
                uncond_vec = self._encode_condition(uncond, B, device)
                # batch-double: single forward for cond/uncond to keep KV cache aligned
                inp_b = torch.cat([inp, inp], dim=0)
                cond_b = torch.cat([cond_vec, uncond_vec], dim=0)
                h = self.decoder(inp_b, cond_b, attn_bias=None)
                h = self.decoder.apply_head_norm(h, cond_b)
                h_cond, h_uncond = h[:B], h[B:]
            pred = (1 + self.cfg.cfg_scale) * self.head(h_cond) - self.cfg.cfg_scale * self.head(h_uncond)

            img_k = unpatchify(pred, self.p, sk, sk)
            if self.apply_conv:
                img_k = self.post_unpatch_convs[k](img_k)

            offset += self.block_sizes[k]

        if self.cfg.decoder_type == "var":
            self.decoder.enable_kv_cache(False)
        return img_k
    

    @torch.no_grad()
    def generate_all_scales(
        self,
        B: int,
        low_sc: torch.Tensor,  # [B, 3, s1, s1], clean lowest-scale image
        labels: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Returns all scales of the generated image, as a list [imgs_s1, ..., imgs_sK] where each imgs_sk is of shape [B, 3, sK, sK].
        Uses KV caching; no blockwise mask needed at inference.
        """

        all_scales = []

        device = next(self.parameters()).device
        if labels is None:
            labels = torch.randint(0, self.cfg.num_classes, (B,), device=device)
        uncond = torch.full((B,), self.cfg.num_classes, device=device, dtype=torch.long)

        # build pos/scale once
        pos = self.pos_1L.to(device=device)
        scale_ids = []
        for k, n in enumerate(self.block_sizes):
            scale_ids.extend([k] * n)
        scale_ids = torch.tensor(scale_ids, device=device)
        scale = self.scale_embed(scale_ids).unsqueeze(0)

        def add_pos(x, offset):
            # If you still want scale embedding, keep "+ scale[...]"
            return x + pos[:, offset:offset + x.shape[1]] + scale[:, offset:offset + x.shape[1]]
            # If you want ONLY positional embedding, drop the "+ scale[...]"

        # first step: predict scale 1
        offset = 0
        n1_block = self.block_sizes[0]

        if self.cfg.decoder_type == "var":
            # Keep prefix context via KV cache across scale blocks
            self.decoder.enable_kv_cache(True)

        # class conditioning only for first block
        start_cond = self._encode_condition(labels, B, device).unsqueeze(1)
        start_uncond = self._encode_condition(uncond, B, device).unsqueeze(1)


        # noisy lowest-scale image token (shared between cond/uncond)
        s1 = self.scales[0]
        #low = torch.zeros(B, 3, s1, s1, device=device)
        low = low_sc.to(device=device)
        low = self._build_first_scale(low)
        low_patches = patchify(low, self.p)
        low_block = self.patch_embed(low_patches)

        block_cond = torch.cat([start_cond, low_block], dim=1)
        block_uncond = torch.cat([start_uncond, low_block], dim=1)

        inp_cond = add_pos(block_cond, offset)
        inp_uncond = add_pos(block_uncond, offset)

        if self.cfg.decoder_type == "gpt2":
            past = None
            inp_b = torch.cat([inp_cond, inp_uncond], dim=0)
            out = self.decoder(inputs_embeds=inp_b, use_cache=True, past_key_values=past)
            h = out.last_hidden_state
            h_cond, h_uncond = h[:B], h[B:]
            past = out.past_key_values
        else:
            cond_vec = self._encode_condition(labels, B, device)
            uncond_vec = self._encode_condition(uncond, B, device)
            # batch-double: single forward for cond/uncond to keep KV cache aligned
            inp = torch.cat([inp_cond, inp_uncond], dim=0)
            cond = torch.cat([cond_vec, uncond_vec], dim=0)
            h = self.decoder(inp, cond, attn_bias=None)
            h = self.decoder.apply_head_norm(h, cond)
            h_cond, h_uncond = h[:B], h[B:]

        pred = (1 + self.cfg.cfg_scale) * self.head(h_cond) - self.cfg.cfg_scale * self.head(h_uncond)
        pred = pred[:, 1:, :]

        # reconstruct scale-1 image
        img_k = unpatchify(pred, self.p, s1, s1)
        if self.apply_conv:
            img_k = self.post_unpatch_convs[0](img_k)
        all_scales.append(img_k)

        # subsequent scales
        offset += n1_block
        for k in range(1, self.num_scales):
            sk = self.scales[k]
            up = F.interpolate(img_k, size=(sk, sk), mode="bicubic", align_corners=False)
            patches = patchify(up, self.p)
            inp = self.patch_embed(patches)

            inp = add_pos(inp, offset)

            if self.cfg.decoder_type == "gpt2":
                # adding class embedding at each scale to match CFG design
                inp_cond = inp + self.class_embed(labels).unsqueeze(1)
                inp_uncond = inp + self.class_embed(uncond).unsqueeze(1)
                inp_b = torch.cat([inp_cond, inp_uncond], dim=0)
                out = self.decoder(inputs_embeds=inp_b, use_cache=True, past_key_values=past)
                h = out.last_hidden_state
                h_cond, h_uncond = h[:B], h[B:]
                past = out.past_key_values
            else:
                cond_vec = self._encode_condition(labels, B, device)
                uncond_vec = self._encode_condition(uncond, B, device)
                # batch-double: single forward for cond/uncond to keep KV cache aligned
                inp_b = torch.cat([inp, inp], dim=0)
                cond_b = torch.cat([cond_vec, uncond_vec], dim=0)
                h = self.decoder(inp_b, cond_b, attn_bias=None)
                h = self.decoder.apply_head_norm(h, cond_b)
                h_cond, h_uncond = h[:B], h[B:]
            pred = (1 + self.cfg.cfg_scale) * self.head(h_cond) - self.cfg.cfg_scale * self.head(h_uncond)

            img_k = unpatchify(pred, self.p, sk, sk)
            if self.apply_conv:
                img_k = self.post_unpatch_convs[k](img_k)
            all_scales.append(img_k)

            offset += self.block_sizes[k]

        if self.cfg.decoder_type == "var":
            self.decoder.enable_kv_cache(False)
        return all_scales
