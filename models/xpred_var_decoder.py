import math
from typing import Optional

import torch
import torch.nn as nn

from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn


class XPredVARDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
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

        norm_layer = lambda dim: nn.LayerNorm(dim, eps=norm_eps)
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
