import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_

try:
    from xformers.ops import memory_efficient_attention
    use_xformer = True
except ImportError:
    use_xformer = False

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)

class FlexVAR(nn.Module):
    def __init__(
        self, vae_local,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1, token_dropout_p=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        self.tok_dropout = nn.Dropout(token_dropout_p)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        self.in_norm = nn.LayerNorm(self.C, eps=norm_eps)

        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        self.pos_embeds = nn.ParameterDict()
        for pn in self.patch_nums:
            pos_2d = self.generate_2d_rotary_position_embedding(h=pn, w=pn, d=self.C)
            self.pos_embeds[str(pn)] = nn.Parameter(pos_2d.reshape(1, self.C, pn*pn).permute(0, 2, 1))
        
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)

        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)
        attn_bias_for_masking = torch.where(d == dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        
        self.heads = nn.ModuleDict({
            str(pn): nn.Linear(self.C, self.V) for pn in self.patch_nums
        })

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def generate_2d_rotary_position_embedding(self, h, w, d):
        assert d % 2 == 0
        pos_encoding = torch.zeros(h, w, d)
        y_coords = torch.arange(h, dtype=torch.float32)
        x_coords = torch.arange(w, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords)
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * -(math.log(10000.0) / d))
        for i in range(h):
            for j in range(w):
                pos_encoding[i, j, 0::2] = torch.sin(y_grid[i, j] * div_term)
                pos_encoding[i, j, 1::2] = torch.cos(x_grid[i, j] * div_term)
        return pos_encoding.unsqueeze(0).permute(0,3,1,2)

    def get_pos_embed(self, patch_nums=None, si=None):
        if patch_nums is None:
            patch_nums = self.patch_nums
        device = self.word_embed.weight.device
        if si is not None:
            pn = patch_nums[si]
            if str(pn) not in self.pos_embeds:
                pos_2d = self.generate_2d_rotary_position_embedding(h=pn, w=pn, d=self.C).to(device)
                self.pos_embeds[str(pn)] = nn.Parameter(pos_2d.reshape(1, self.C, pn*pn).permute(0, 2, 1))
            return self.pos_embeds[str(pn)]
        pos_embeddings = []
        for i, pn in enumerate(patch_nums):
            if str(pn) not in self.pos_embeds:
                pos_2d = self.generate_2d_rotary_position_embedding(h=pn, w=pn, d=self.C).to(device)
                self.pos_embeds[str(pn)] = nn.Parameter(pos_2d.reshape(1, self.C, pn*pn).permute(0, 2, 1))
            pos_embeddings.append(self.pos_embeds[str(pn)])
        return torch.cat(pos_embeddings, dim=1)

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor], infer_pn: Optional[int] = None):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual
            h = resi + self.blocks[-1].drop_path(h)
        else:
            h = h_or_h_and_residual
            
        h = self.head_nm(h, cond_BD)
        
        if infer_pn is not None:
            idx = self.patch_nums.index(infer_pn)
            active_dim = (idx + 1) * (self.C // len(self.patch_nums))
            h_masked = h.clone()
            h_masked[:, :, active_dim:] = 0.0
            return self.heads[str(infer_pn)](h_masked)
            
        all_logits = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            seq_len = pn ** 2
            h_slice = h[:, cur : cur + seq_len, :]
            
            active_dim = (i + 1) * (self.C // len(self.patch_nums))
            h_masked = h_slice.clone()
            h_masked[:, :, active_dim:] = 0.0
            
            all_logits.append(self.heads[str(pn)](h_masked))
            cur += seq_len
        return torch.cat(all_logits, dim=1)

    def _get_attn_bias_with_class_token(self, dtype):
        L = self.L
        device = self.attn_bias_for_masking.device
        total = L + 1

        bias = torch.full((1, 1, total, total), -torch.inf, dtype=dtype, device=device)
        bias[:, :, 1:, 1:] = self.attn_bias_for_masking.to(dtype)
        bias[:, :, 1:, 0] = 0.
        bias[:, :, 0, 0]  = 0.

        if use_xformer:
            total_padded = (total + 7) // 8 * 8
            if total_padded != total:
                bias_padded = torch.full((1, 1, total_padded, total_padded), -torch.inf, dtype=dtype, device=device)
                bias_padded[:, :, :total, :total] = bias
                bias_padded[:, :, total:, total:] = 0.
                bias = bias_padded

        return bias

    def update_patch_related(self, infer_patch_nums):
        self.patch_nums = infer_patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        cur = 0
        self.begin_ends = []
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
            if str(pn) not in self.heads:
                self.heads[str(pn)] = nn.Linear(self.C, self.V).to(self.word_embed.weight.device)
        self.num_stages_minus_1 = len(self.patch_nums) - 1

        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)
        attn_bias_for_masking = torch.where(d == dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        if use_xformer:
            b, c, l, l = attn_bias_for_masking.shape
            l_ = (l + 7) // 8 * 8
            attn_bias_for_masking_ = torch.full((b, c, l_, l_), -torch.inf)
            attn_bias_for_masking_[:, :, :l, :l] = attn_bias_for_masking
            attn_bias_for_masking_[:,:,l:, l:] = 0
            attn_bias_for_masking = attn_bias_for_masking_
        self.attn_bias_for_masking = attn_bias_for_masking.to(self.attn_bias_for_masking.device)

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, vqvae, B: int, label_B: Optional[Union[int, torch.LongTensor]], infer_patch_nums,
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False, max_pn=16, used_llamagen_cfg=False, invalid_ids=None,
    ) -> torch.Tensor:
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng

        if list(infer_patch_nums) != list(self.patch_nums):
            self.update_patch_related(infer_patch_nums)

        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.class_emb.weight.device)

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))

        class_token = sos.unsqueeze(1)

        scale_0_dim = self.C // len(infer_patch_nums)
        lvl_pos_0 = self.get_pos_embed(patch_nums=infer_patch_nums, si=0)
        scale_tokens_0 = (
            sos.unsqueeze(1).expand(2*B, self.first_l, -1)
            + self.pos_start.expand(2*B, self.first_l, -1)
            + lvl_pos_0
        )
        
        mask_0 = torch.zeros_like(scale_tokens_0)
        mask_0[:, :, :scale_0_dim] = 1.0
        scale_tokens_0 = scale_tokens_0 * mask_0
        
        next_token_map = torch.cat([class_token, scale_tokens_0], dim=1)

        vae_embedding = F.normalize(vqvae.quantize.embedding.weight, p=2, dim=-1)

        for si, pn in enumerate(infer_patch_nums):
            ratio = si / min(len(infer_patch_nums)-1, 9)
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)

            x = next_token_map
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)

            logits_BlV = self.get_logits(x[:, 1:], cond_BD, infer_pn=pn)

            if invalid_ids is not None:
                logits_BlV[:, :, invalid_ids] = -100.0

            if not used_llamagen_cfg:
                t = cfg * ratio if pn <= max_pn else cfg
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            else:
                cond_logits, uncond_logits = torch.split(logits_BlV, len(logits_BlV) // 2, dim=0)
                logits_BlV = uncond_logits + (cond_logits - uncond_logits) * cfg

            if pn > max_pn or pn not in [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 23, 24, 32]:
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=top_p, num_samples=1)[:, :, 0]
            else:
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

            assert not more_smooth
            if si != len(infer_patch_nums)-1:
                next_hw = infer_patch_nums[si+1]
                curr_hw = infer_patch_nums[si]

                curr_quant = vae_embedding[idx_Bl].reshape(B, curr_hw, curr_hw, vae_embedding.shape[-1]).permute(0,3,1,2)
                next_quant = F.interpolate(curr_quant, size=(next_hw, next_hw), mode='bicubic', align_corners=False)
                next_quant = next_quant.reshape(B, curr_quant.shape[1], -1).permute(0,2,1)

                scale_tokens_next = self.word_embed(next_quant)
                scale_tokens_next = self.in_norm(scale_tokens_next)
                scale_tokens_next = scale_tokens_next + self.get_pos_embed(patch_nums=infer_patch_nums, si=si+1)
                
                active_dim = (si + 2) * (self.C // len(infer_patch_nums))
                mask_next = torch.zeros_like(scale_tokens_next)
                mask_next[:, :, :active_dim] = 1.0
                scale_tokens_next = scale_tokens_next * mask_next
                
                scale_tokens_next = scale_tokens_next.repeat(2, 1, 1)

                next_token_map = torch.cat([class_token, scale_tokens_next], dim=1)

        last_pn = infer_patch_nums[-1]
        z_shape = (B, vqvae.Cvae, last_pn, last_pn)
        return vqvae.decode_code(idx_Bl, shape=z_shape)

    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor, infer_patch_nums) -> torch.Tensor:
        if list(infer_patch_nums) != list(self.patch_nums):
            self.update_patch_related(infer_patch_nums)

        B = x_BLCv_wo_first_l.shape[0]

        label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
        sos = cond_BD = self.class_emb(label_B)

        class_token = sos.unsqueeze(1)
        
        scale_0_dim = self.C // len(self.patch_nums)
        scale_tokens_0 = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
        
        mask_0 = torch.zeros_like(scale_tokens_0)
        mask_0[:, :, :scale_0_dim] = 1.0
        scale_tokens_0 = scale_tokens_0 * mask_0

        if self.prog_si == 0:
            x_BLC = torch.cat([class_token, scale_tokens_0], dim=1)
        else:
            embedded_x = self.word_embed(x_BLCv_wo_first_l.to(sos.dtype))
            embedded_x = self.in_norm(embedded_x)
            embedded_x = self.tok_dropout(embedded_x)
            
            cur = 0
            for i, pn in enumerate(self.patch_nums):
                if i == 0: continue
                
                length = pn ** 2
                active_dim = (i + 1) * (self.C // len(self.patch_nums))
                
                start_idx = cur
                end_idx = cur + length
                
                scale_slice = embedded_x[:, start_idx:end_idx, :]
                mask = torch.zeros_like(scale_slice)
                mask[:, :, :active_dim] = 1.0
                embedded_x[:, start_idx:end_idx, :] = scale_slice * mask
                
                cur += length
            
            x_BLC = torch.cat([
                class_token,
                scale_tokens_0,
                embedded_x
            ], dim=1)

        lvl_pos = self.get_pos_embed(patch_nums=self.patch_nums, si=None)
        
        lvl_pos_masked = lvl_pos[:, :self.L].clone()
        cur_pos = 0
        for i, pn in enumerate(self.patch_nums):
            length = pn ** 2
            active_dim = (i + 1) * (self.C // len(self.patch_nums))
            lvl_pos_masked[:, cur_pos:cur_pos+length, active_dim:] = 0.0
            cur_pos += length
            
        x_BLC[:, 1:] = x_BLC[:, 1:] + lvl_pos_masked

        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        attn_bias = self._get_attn_bias_with_class_token(dtype=x_BLC.dtype)

        for b in self.blocks:
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)

        x_BLC = self.get_logits(x_BLC[:, 1:], cond_BD)
        return x_BLC

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        if init_head >= 0:
            for i, pn in enumerate(self.patch_nums):
                head = self.heads[str(pn)]
                active_dim = (i + 1) * (self.C // len(self.patch_nums))
                head.weight.data.mul_(init_head)
                head.weight.data[:, active_dim:] = 0.0
                head.bias.data.zero_()
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'