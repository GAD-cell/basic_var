from typing import Optional, Tuple, List
from dataclasses import dataclass
from transformers import GPT2Config, GPT2Model

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn


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
    decoder_type: str = "gpt2"       # "gpt2" or "var"
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0
    attn_l2_norm: bool = False
    shared_aln: bool = False
    num_classes: int = 1000
    cond_drop_prob: float = 0.1
    first_scale_noise_std: float = 0.0


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

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
        # scale 1: class/pos/scale token + noisy lowest-scale image
        n1 = self.block_sizes[0]
        start = self._encode_condition(labels, B, device).unsqueeze(1).expand(-1, n1, -1)

        low = imgs_k[0]
        low = low + torch.randn_like(low) * self.cfg.first_scale_noise_std
        low_patches = patchify(low, self.p)
        low_block = self.patch_embed(low_patches)
        start = start + low_block
        blocks.append(start)

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
        return F.mse_loss(preds, targets)

    @torch.no_grad()
    def generate(
        self,
        B: int,
        labels: Optional[torch.Tensor] = None,
        cfg_scale: float = 0.7,
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
        n1 = self.block_sizes[0]

        # class conditioning only for first block
        start_cond = self._encode_condition(labels, B, device).unsqueeze(1).expand(-1, n1, -1)
        start_uncond = self._encode_condition(uncond, B, device).unsqueeze(1).expand(-1, n1, -1)


        # noisy lowest-scale image token (shared between cond/uncond)
        s1 = self.scales[0]
        low = torch.zeros(B, 3, s1, s1, device=device)
        if self.cfg.first_scale_noise_std > 0:
            low = low + torch.randn_like(low) * self.cfg.first_scale_noise_std
        low_patches = patchify(low, self.p)
        low_block = self.patch_embed(low_patches)

        start_cond = start_cond + low_block
        start_uncond = start_uncond + low_block

        inp_cond = add_pos(start_cond, offset)
        inp_uncond = add_pos(start_uncond, offset)

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
        img_k = unpatchify(pred, self.p, s1, s1)

        # subsequent scales
        offset += n1
        for k in range(1, self.num_scales):
            sk = self.scales[k]
            up = F.interpolate(img_k, size=(sk, sk), mode="bicubic", align_corners=False)
            patches = patchify(up, self.p)
            inp = self.patch_embed(patches)

            inp = add_pos(inp, offset)

            if self.cfg.decoder_type == "gpt2":
                # adding class embedding at each scale to match CFG design
                inp_cond = inp_cond + self.class_embed(labels).unsqueeze(1)
                inp_uncond = inp_uncond + self.class_embed(uncond).unsqueeze(1)
                out_cond = self.decoder(inputs_embeds=inp_cond, use_cache=True, past_key_values=past)
                out_uncond = self.decoder(inputs_embeds=inp_uncond, use_cache=True, past_key_values=past)
                h_cond = out_cond.last_hidden_state
                h_uncond = out_uncond.last_hidden_state
                past = out_cond.past_key_values
            else:
                cond_vec = self._encode_condition(labels, B, device)
                uncond_vec = self._encode_condition(uncond, B, device)
                # no explicit class embedding in the input (following original VAR implementation)
                h_cond = self.decoder(inp, cond_vec, attn_bias=None)
                h_uncond = self.decoder(inp, uncond_vec, attn_bias=None)
                h_cond = self.decoder.apply_head_norm(h_cond, cond_vec)
                h_uncond = self.decoder.apply_head_norm(h_uncond, uncond_vec)
            pred = (1 + cfg_scale) * self.head(h_cond) - cfg_scale * self.head(h_uncond)

            img_k = unpatchify(pred, self.p, sk, sk)

            offset += self.block_sizes[k]

        if self.cfg.decoder_type == "var":
            self.decoder.enable_kv_cache(False)
        return img_k
