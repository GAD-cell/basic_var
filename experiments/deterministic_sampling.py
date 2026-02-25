"""
Deterministic sampling helpers for XPredNextScale.

This implements a generate-like function that:
1) Samples B images from the training set with labels.
2) Downsamples them to the lowest scale using area pooling.
3) Uses the sampled lowest-scale image to condition generation of higher scales.
4) Does not inject any noise (no noise seed, no input noise).
"""

from typing import Optional
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.xpred_var_decoder import XPredConfig, XPredNextScale, patchify, unpatchify


@torch.no_grad()
def generate_from_dataset_lowest_scale(
    model,
    train_ds,
    B: int,
    cfg_scale: float = 0.7,
    device: Optional[torch.device] = None,
):
    """
    Deterministic generation conditioned on a sampled lowest-scale image.

    Args:
      model: XPredNextScale
      train_ds: dataset returning (img, label)
      B: number of images to generate
      cfg_scale: classifier-free guidance scale
      device: optional torch.device

    Returns:
      Tensor [B, 3, sK, sK]
    """
    model.eval()
    device = device or next(model.parameters()).device

    # Sample B items from dataset
    idx = torch.randint(0, len(train_ds), (B,))
    imgs = []
    labels = []
    for i in idx.tolist():
        img, lab = train_ds[i]
        imgs.append(img)
        labels.append(lab)
    imgs = torch.stack(imgs, dim=0).to(device)
    labels = torch.tensor(labels, device=device, dtype=torch.long)

    # Downsample to lowest scale using area pooling
    s1 = model.scales[0]
    img_k = F.interpolate(imgs, size=(s1, s1), mode="area")

    # Prep conditioning
    uncond = torch.full((B,), model.cfg.num_classes, device=device, dtype=torch.long)
    pos = model.pos_1L.to(device=device)
    scale_ids = []
    for k, n in enumerate(model.block_sizes):
        scale_ids.extend([k] * n)
    scale_ids = torch.tensor(scale_ids, device=device)
    scale = model.scale_embed(scale_ids).unsqueeze(0)

    def add_pos(x, offset):
        return x + pos[:, offset:offset + x.shape[1]] + scale[:, offset:offset + x.shape[1]]

    # First step: run start token to initialize cache for GPT2 (prediction ignored)
    if model.cfg.decoder_type == "var":
        model.decoder.enable_kv_cache(True)

    offset = 0
    start_cond = model._encode_condition(labels, B, device).unsqueeze(1)
    start_uncond = model._encode_condition(uncond, B, device).unsqueeze(1)

    low_patches = patchify(img_k, model.p)
    low_block = model.patch_embed(low_patches)
    block_cond = torch.cat([start_cond, low_block], dim=1)
    block_uncond = torch.cat([start_uncond, low_block], dim=1)

    inp_cond = add_pos(block_cond, offset)
    inp_uncond = add_pos(block_uncond, offset)

    if model.cfg.decoder_type == "gpt2":
        past = None
        out_cond = model.decoder(inputs_embeds=inp_cond, use_cache=True, past_key_values=past)
        out_uncond = model.decoder(inputs_embeds=inp_uncond, use_cache=True, past_key_values=past)
        h_cond = out_cond.last_hidden_state
        h_uncond = out_uncond.last_hidden_state
        past = out_cond.past_key_values
    else:
        cond_vec = model._encode_condition(labels, B, device)
        uncond_vec = model._encode_condition(uncond, B, device)
        h_cond = model.decoder(inp_cond, cond_vec, attn_bias=None)
        h_uncond = model.decoder(inp_uncond, uncond_vec, attn_bias=None)
        h_cond = model.decoder.apply_head_norm(h_cond, cond_vec)
        h_uncond = model.decoder.apply_head_norm(h_uncond, uncond_vec)
        past = None

    _ = (1 + cfg_scale) * model.head(h_cond) - cfg_scale * model.head(h_uncond)

    # Subsequent scales conditioned on sampled lowest-scale image
    offset += model.block_sizes[0]
    for k in range(1, model.num_scales):
        sk = model.scales[k]
        up = F.interpolate(img_k, size=(sk, sk), mode="bicubic", align_corners=False)
        patches = patchify(up, model.p)
        inp = model.patch_embed(patches)
        inp = add_pos(inp, offset)
        if model.cfg.decoder_type == "gpt2":
            inp_cond = inp + model.class_embed(labels).unsqueeze(1)
            inp_uncond = inp + model.class_embed(uncond).unsqueeze(1)

        if model.cfg.decoder_type == "gpt2":
            out_cond = model.decoder(inputs_embeds=inp_cond, use_cache=True, past_key_values=past)
            out_uncond = model.decoder(inputs_embeds=inp_uncond, use_cache=True, past_key_values=past)
            h_cond = out_cond.last_hidden_state
            h_uncond = out_uncond.last_hidden_state
            past = out_cond.past_key_values
        else:
            cond_vec = model._encode_condition(labels, B, device)
            uncond_vec = model._encode_condition(uncond, B, device)
            h_cond = model.decoder(inp, cond_vec, attn_bias=None)
            h_uncond = model.decoder(inp, uncond_vec, attn_bias=None)
            h_cond = model.decoder.apply_head_norm(h_cond, cond_vec)
            h_uncond = model.decoder.apply_head_norm(h_uncond, uncond_vec)

        pred = (1 + cfg_scale) * model.head(h_cond) - cfg_scale * model.head(h_uncond)
        img_k = unpatchify(pred, model.p, sk, sk)
        offset += model.block_sizes[k]

    if model.cfg.decoder_type == "var":
        model.decoder.enable_kv_cache(False)

    return imgs, img_k

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def generate_from_cifar10_lowest_scale_and_save():
    ckpt_file : str = "checkpoints/step_760000-deterministic_mse.pt"
    B = 4
    cfg_scale = 0.7

    device = pick_device()
    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms.ToTensor())
    ckpt = torch.load(ckpt_file, map_location=device)
    cfg = XPredConfig(**ckpt["model_cfg"])
    model = XPredNextScale(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        o_imgs, gen_imgs = generate_from_dataset_lowest_scale(model, dataset, B, cfg_scale=cfg_scale, device=device)
    
    print("Saving generated images to deterministic_sampling.png ...")
    # Save mosaic of original and generated images
    o_imgs = o_imgs.cpu().permute(0, 2, 3, 1).numpy()  # B, H, W, C
    gen_imgs = gen_imgs.cpu().permute(0, 2, 3, 1).numpy()
    mosaic = np.concatenate([np.concatenate([o, g], axis=1) for o, g in zip(o_imgs, gen_imgs)], axis=0)
    mosaic = np.clip(mosaic, 0.0, 1.0)

    ckpt_filename = Path(ckpt_file).name
    plt.imsave(f"experiments/output/deterministic_sampling_{ckpt_filename}.png", mosaic)

if __name__ == "__main__":
    generate_from_cifar10_lowest_scale_and_save()
