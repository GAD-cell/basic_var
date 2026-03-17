import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms
from datasets import load_dataset

from models.flexvar import FlexVAR
from models.vq_llama import VQ_8

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_NUMS  = [1, 2, 4, 6, 8, 10, 13, 16]
BATCH_SIZE  = 8
VQ_PATH     = "checkpoints_vae/vqvae_epoch_10.pth"
FLEXVAR_PATH= "checkpoints_vae/flexvar_mat.pth"
OUT_DIR     = "debug_vis"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def decode_indices(vqvae, indices, pn, B):
    """Décode des indices (B, pn²) → image (B, 3, H, W) en [0,1]."""
    shape = (B, vqvae.Cvae, pn, pn)
    img = vqvae.decode_code(indices, shape=shape)
    return torch.clamp((img + 1.0) / 2.0, 0, 1)

def to_np(t):
    return t.detach().cpu().permute(1, 2, 0).numpy()

# ── Chargement modèles ─────────────────────────────────────────────────────────
print("Chargement des modèles...")
vqvae = VQ_8().to(DEVICE)
vqvae.load_state_dict(torch.load(VQ_PATH, map_location=DEVICE)["model_state_dict"])
vqvae.eval()
for p in vqvae.parameters():
    p.requires_grad = False

model = FlexVAR(
    vae_local=vqvae,
    num_classes=10,
    depth=12,
    embed_dim=512,
    num_heads=8,
    patch_nums=PATCH_NUMS,
    attn_l2_norm=True,
).to(DEVICE)

ckpt = torch.load(FLEXVAR_PATH, map_location=DEVICE)
state = ckpt["model"] if "model" in ckpt else ckpt
model.load_state_dict(state, strict=False)
model.eval()
print("Modèles chargés.")

# ── Données ────────────────────────────────────────────────────────────────────
print("Chargement CIFAR-10...")
raw = load_dataset("cifar10", split="train")
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

imgs, labels = [], []
for i in range(BATCH_SIZE):
    item = raw[i]
    img = item["img"]
    if img.mode != "RGB":
        img = img.convert("RGB")
    imgs.append(transform(img))
    labels.append(item["label"])

imgs_t   = torch.stack(imgs).to(DEVICE)    # (B,3,32,32) en [-1,1]
label_b  = torch.tensor(labels).to(DEVICE)
B = imgs_t.shape[0]

CIFAR_CLASSES = ["airplane","automobile","bird","cat","deer",
                 "dog","frog","horse","ship","truck"]

# ── Preprocessing identique au dataset précomputé ──────────────────────────────
print("Encodage VAE multi-échelles...")
vae_embedding = F.normalize(vqvae.quantize.embedding.weight, p=2, dim=-1)

with torch.no_grad():
    h = vqvae.encode_conti(imgs_t)

    batch_indices, batch_inputs = [], []
    for j, curr_hw in enumerate(PATCH_NUMS):
        _h = F.interpolate(h, size=(curr_hw, curr_hw), mode='area')
        _, _, log = vqvae.quantize(_h)
        idx = log[-1].view(B, -1)
        batch_indices.append(idx)

        if j < len(PATCH_NUMS) - 1:
            next_hw = PATCH_NUMS[j + 1]
            curr_quant = vae_embedding[idx].reshape(B, curr_hw, curr_hw, -1).permute(0,3,1,2)
            next_quant = F.interpolate(curr_quant, size=(next_hw, next_hw), mode='bicubic')
            next_quant = next_quant.reshape(B, -1, next_hw*next_hw).permute(0,2,1)
            batch_inputs.append(next_quant)

    target_indices  = torch.cat(batch_indices, dim=1)   # (B, L)
    x_input_quants  = torch.cat(batch_inputs,  dim=1)   # (B, L-last_pn², Cvae)

# ── Inférence ──────────────────────────────────────────────────────────────────
print("Inférence FlexVAR...")
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    logits = model(label_b, x_input_quants, PATCH_NUMS)   # (B, L, V)
    preds  = logits.argmax(dim=-1)                         # (B, L)

# ── Visualisation ──────────────────────────────────────────────────────────────
FINAL_SIZE = 64   # taille d'affichage uniforme pour toutes les échelles

print("Génération des figures...")
for b in range(B):
    n_scales = len(PATCH_NUMS)
    # Layout : 3 lignes (GT / Pred / diff) × n_scales colonnes + 1 col image originale
    fig = plt.figure(figsize=(2.5 * (n_scales + 1), 8))
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(
        3, n_scales + 1,
        hspace=0.05, wspace=0.05,
        left=0.02, right=0.98, top=0.88, bottom=0.02
    )

    label_str = CIFAR_CLASSES[labels[b]]
    fig.suptitle(f"Sample {b} — classe : {label_str}", fontsize=13,
                 fontweight='bold', color='#1a1a1a', y=0.95)

    # ── Colonne 0 : image originale ──
    ax_orig = fig.add_subplot(gs[:, 0])
    orig_np = torch.clamp((imgs_t[b] + 1.0) / 2.0, 0, 1).cpu().permute(1,2,0).numpy()
    ax_orig.imshow(orig_np, interpolation='nearest')
    ax_orig.set_title("Original\n32×32", fontsize=8, fontweight='bold', color='#333')
    ax_orig.axis('off')

    # ── Colonnes 1..n : une par scale ──
    cur = 0
    for si, pn in enumerate(PATCH_NUMS):
        seq_len = pn ** 2

        gt_idx   = target_indices[b:b+1, cur:cur+seq_len]   # (1, pn²)
        pred_idx = preds[b:b+1, cur:cur+seq_len]             # (1, pn²)

        with torch.no_grad():
            gt_img   = decode_indices(vqvae, gt_idx,   pn, 1)[0]  # (3,pn,pn)
            pred_img = decode_indices(vqvae, pred_idx, pn, 1)[0]

        # Upscale pour affichage
        def up(t):
            return F.interpolate(t.unsqueeze(0), size=(FINAL_SIZE, FINAL_SIZE),
                                 mode='nearest')[0].cpu().permute(1,2,0).numpy()

        gt_np   = up(gt_img)
        pred_np = up(pred_img)
        diff_np = np.abs(gt_np - pred_np)

        # Accuracy de cette scale pour ce sample
        acc = (pred_idx == gt_idx).float().mean().item() * 100
        n_unique_pred = len(torch.unique(pred_idx))
        n_unique_gt   = len(torch.unique(gt_idx))

        col = si + 1

        # Ligne 0 : GT
        ax_gt = fig.add_subplot(gs[0, col])
        ax_gt.imshow(gt_np, interpolation='nearest')
        ax_gt.axis('off')
        ax_gt.set_title(f"{pn}×{pn}", fontsize=8, fontweight='bold', color='#1a1a1a', pad=3)
        if si == 0:
            ax_gt.set_ylabel("GT", fontsize=9, color='#2A7FBF', fontweight='bold',
                             rotation=0, labelpad=28, va='center')

        # Ligne 1 : Pred
        ax_pred = fig.add_subplot(gs[1, col])
        ax_pred.imshow(pred_np, interpolation='nearest')
        ax_pred.axis('off')
        if si == 0:
            ax_pred.set_ylabel("Pred", fontsize=9, color='#D94F37', fontweight='bold',
                               rotation=0, labelpad=28, va='center')

        # Ligne 2 : métriques texte
        ax_txt = fig.add_subplot(gs[2, col])
        ax_txt.axis('off')
        color_acc = '#2BAA72' if acc > 20 else ('#E8A020' if acc > 5 else '#D94F37')
        ax_txt.text(0.5, 0.6, f"{acc:.1f}%", ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color_acc,
                    transform=ax_txt.transAxes)
        ax_txt.text(0.5, 0.2,
                    f"u:{n_unique_pred}/{n_unique_gt}",
                    ha='center', va='center', fontsize=6.5, color='#888',
                    transform=ax_txt.transAxes)

        cur += seq_len

    out_path = os.path.join(OUT_DIR, f"sample_{b:02d}_{label_str}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Sauvegardé : {out_path}")

# ── Résumé global ──────────────────────────────────────────────────────────────
print("\n===== Résumé précision par scale (moyenne sur le batch) =====")
cur = 0
for pn in PATCH_NUMS:
    seq_len = pn ** 2
    gt_s   = target_indices[:, cur:cur+seq_len]
    pred_s = preds[:, cur:cur+seq_len]
    acc    = (pred_s == gt_s).float().mean().item() * 100
    n_uniq = torch.unique(pred_s).numel()
    print(f"  {pn:>2}×{pn:<2}  acc={acc:5.1f}%  tokens_uniques_prédits={n_uniq}/1024")
    cur += seq_len

print(f"\nFigures sauvegardées dans ./{OUT_DIR}/")

# ── Génération partielle autorégresssive ──────────────────────────────────────
from models.helpers import sample_with_top_k_top_p_

@torch.no_grad()
def partial_autoregressive_infer(model, vqvae, label_B, gt_indices, start_si, patch_nums, B, device,
                                  cfg=3.0, top_k=30, top_p=0.9):
    """
    Génère une image en forçant les indices GT jusqu'à start_si (inclus),
    puis laisse le modèle générer autorégressivement les scales suivantes.
    Retourne (image_finale, all_indices (B, L))
    """
    vae_emb = F.normalize(vqvae.quantize.embedding.weight, p=2, dim=-1)

    sos = cond_BD = model.class_emb(
        torch.cat((label_B, torch.full_like(label_B, fill_value=model.num_classes)), dim=0)
    )
    class_token = sos.unsqueeze(1)

    scale_0_dim  = model.C // len(patch_nums)
    lvl_pos_0    = model.get_pos_embed(patch_nums=patch_nums, si=0)
    scale_tokens_0 = (
        sos.unsqueeze(1).expand(2*B, model.first_l, -1)
        + model.pos_start.expand(2*B, model.first_l, -1)
        + lvl_pos_0
    )
    mask_0 = torch.zeros_like(scale_tokens_0)
    mask_0[:, :, :scale_0_dim] = 1.0
    scale_tokens_0  = scale_tokens_0 * mask_0
    next_token_map  = torch.cat([class_token, scale_tokens_0], dim=1)

    cond_BD_or_gss  = model.shared_ada_lin(cond_BD)
    all_indices     = []
    idx_Bl          = None

    for si, pn in enumerate(patch_nums):
        ratio = si / min(len(patch_nums) - 1, 9)

        x = next_token_map
        for blk in model.blocks:
            x = blk(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)

        logits_BlV = model.get_logits(x[:, 1:], cond_BD, infer_pn=pn)
        t          = cfg * ratio
        logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

        if si <= start_si:
            # Forcer les indices GT pour cette scale
            cur    = sum(p**2 for p in patch_nums[:si])
            idx_Bl = gt_indices[:, cur : cur + pn**2]
        else:
            # Laisser le modèle générer
            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV, rng=None, top_k=top_k, top_p=top_p, num_samples=1
            )[:, :, 0]

        all_indices.append(idx_Bl)

        if si != len(patch_nums) - 1:
            next_hw    = patch_nums[si + 1]
            curr_hw    = pn
            curr_quant = vae_emb[idx_Bl].reshape(B, curr_hw, curr_hw, vae_emb.shape[-1]).permute(0,3,1,2)
            next_quant = F.interpolate(curr_quant, size=(next_hw, next_hw), mode='bicubic', align_corners=False)
            next_quant = next_quant.reshape(B, curr_quant.shape[1], -1).permute(0, 2, 1)

            scale_tokens_next = model.word_embed(next_quant)
            scale_tokens_next = model.in_norm(scale_tokens_next)
            scale_tokens_next = scale_tokens_next + model.get_pos_embed(patch_nums=patch_nums, si=si+1)

            active_dim = (si + 2) * (model.C // len(patch_nums))
            mask_next  = torch.zeros_like(scale_tokens_next)
            mask_next[:, :, :active_dim] = 1.0
            scale_tokens_next = scale_tokens_next * mask_next
            scale_tokens_next = scale_tokens_next.repeat(2, 1, 1)

            next_token_map = torch.cat([class_token, scale_tokens_next], dim=1)

    last_pn = patch_nums[-1]
    img_out = vqvae.decode_code(idx_Bl, shape=(B, vqvae.Cvae, last_pn, last_pn))
    img_out = torch.clamp((img_out + 1.0) / 2.0, 0, 1)
    return img_out, torch.cat(all_indices, dim=1)


# ── Visualisation : génération partielle depuis différentes scales ─────────────
print("\nGénération partielle depuis différentes scales de départ...")

START_SCALES = [0, 1, 2, 3]  # 1×1, 2×2, 4×4, 6×6
N_COLS = len(PATCH_NUMS) + 2  # label + GT finale + 8 scales
N_ROWS = len(START_SCALES)

fig = plt.figure(figsize=(2.2 * N_COLS, 2.2 * (N_ROWS + 1)))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(
    N_ROWS + 1, N_COLS,
    hspace=0.08, wspace=0.05,
    left=0.06, right=0.98, top=0.93, bottom=0.02
)
fig.suptitle("Partial generation — GT seed up to scale k, model generates the rest",
             fontsize=12, fontweight='bold', color='#1a1a1a', y=0.97)

# ── Ligne 0 : GT une seule fois ───────────────────────────────────────────────
ax_lbl_gt = fig.add_subplot(gs[0, 0])
ax_lbl_gt.axis('off')
ax_lbl_gt.text(0.5, 0.5, "GT", ha='center', va='center', fontsize=9,
               fontweight='bold', color='#2A7FBF', transform=ax_lbl_gt.transAxes)

last_pn  = PATCH_NUMS[-1]
last_seq = last_pn ** 2
gt_last_idx = target_indices[4:5, -last_seq:]
gt_last_img = decode_indices(vqvae, gt_last_idx, last_pn, 1)[0]

def up(t):
    return F.interpolate(t.unsqueeze(0), size=(FINAL_SIZE, FINAL_SIZE),
                         mode='nearest')[0].cpu().permute(1,2,0).numpy()

gt_np_final = up(gt_last_img)
ax_gt_f = fig.add_subplot(gs[0, 1])
ax_gt_f.imshow(gt_np_final, interpolation='nearest')
ax_gt_f.axis('off')
ax_gt_f.set_title(f"GT\n{last_pn}×{last_pn}", fontsize=7, fontweight='bold', color='#2A7FBF')

cur = 0
for si, pn in enumerate(PATCH_NUMS):
    seq_len  = pn ** 2
    col      = si + 2
    gt_idx_s = target_indices[4:5, cur:cur+seq_len]
    gt_img_s = decode_indices(vqvae, gt_idx_s, pn, 1)[0]
    ax_gt = fig.add_subplot(gs[0, col])
    ax_gt.imshow(up(gt_img_s), interpolation='nearest')
    ax_gt.axis('off')
    ax_gt.set_title(f"{pn}×{pn}", fontsize=7, fontweight='bold', color='#1a1a1a', pad=2)
    for spine in ax_gt.spines.values():
        spine.set_visible(True)
        spine.set_color('#2A7FBF')
        spine.set_linewidth(0.8)
    if col == 2:
        ax_gt.set_ylabel("GT", fontsize=7, color='#2A7FBF',
                         fontweight='bold', rotation=0, labelpad=22, va='center')
    cur += seq_len

# ── Lignes 1..N_ROWS : pred uniquement, une par start_scale ──────────────────
for row_i, start_si in enumerate(START_SCALES):
    start_pn = PATCH_NUMS[start_si]
    row = row_i + 1

    with torch.no_grad():
        _, all_preds_combined = partial_autoregressive_infer(
            model=model, vqvae=vqvae,
            label_B=label_b[:5],
            gt_indices=target_indices[:5],
            start_si=start_si,
            patch_nums=PATCH_NUMS,
            B=5, device=DEVICE,
        )

    # Colonne 0 : label
    ax_lbl = fig.add_subplot(gs[row, 0])
    ax_lbl.axis('off')
    ax_lbl.text(0.5, 0.5, f"Start\n{start_pn}×{start_pn}",
                ha='center', va='center', fontsize=9,
                fontweight='bold', color='#1a1a1a',
                transform=ax_lbl.transAxes)

    # Colonne 1 : image finale générée
    last_pred_idx = all_preds_combined[4:5, -last_seq:]
    last_pred_img = decode_indices(vqvae, last_pred_idx, last_pn, 1)[0]
    ax_final = fig.add_subplot(gs[row, 1])
    ax_final.imshow(up(last_pred_img), interpolation='nearest')
    ax_final.axis('off')

    # Colonnes 2..N : pred par scale
    cur = 0
    for si, pn in enumerate(PATCH_NUMS):
        seq_len    = pn ** 2
        col        = si + 2
        pred_idx_s = all_preds_combined[4:5, cur:cur+seq_len]
        pred_img_s = decode_indices(vqvae, pred_idx_s, pn, 1)[0]

        is_seed  = (si <= start_si)
        is_start = (si == start_si)
        bcol     = '#2A7FBF' if is_seed else '#D94F37'
        lw       = 2.5 if is_start else (1.0 if is_seed else 0.3)

        ax_pred = fig.add_subplot(gs[row, col])
        ax_pred.imshow(up(pred_img_s), interpolation='nearest')
        ax_pred.axis('off')
        for spine in ax_pred.spines.values():
            spine.set_visible(True)
            spine.set_color(bcol)
            spine.set_linewidth(lw)
        if col == 2:
            ax_pred.set_ylabel("Pred", fontsize=7, color='#D94F37',
                               fontweight='bold', rotation=0, labelpad=22, va='center')
        cur += seq_len

out_path = os.path.join(OUT_DIR, "partial_generation_all_scales.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"  Sauvegardé : {out_path}")

# ── Distribution scale 1×1 par classe ─────────────────────────────────────────
print("\n===== Distribution scale 1×1 par classe (KL divergence) =====")
gt_scale0   = target_indices[:, 0]   # (B,) — un token par image
pred_scale0 = preds[:, 0]            # (B,)

for c in range(10):
    mask = (label_b == c)
    if mask.sum() == 0:
        continue
    gt_tokens   = gt_scale0[mask]
    pred_tokens = pred_scale0[mask]

    gt_dist   = torch.bincount(gt_tokens,   minlength=1024).float()
    pred_dist = torch.bincount(pred_tokens, minlength=1024).float()
    gt_dist   /= gt_dist.sum()
    pred_dist /= pred_dist.sum()

    kl = F.kl_div(
        (pred_dist + 1e-8).log(),
        gt_dist + 1e-8,
        reduction='sum'
    ).item()

    top_pred = torch.topk(pred_dist, 3).indices.tolist()
    top_gt   = torch.topk(gt_dist,   3).indices.tolist()

    print(f"  {CIFAR_CLASSES[c]:<12} n={mask.sum().item()}  KL={kl:.3f}  "
          f"top_gt={top_gt}  top_pred={top_pred}")

# ── Grid search FID ───────────────────────────────────────────────────────────
print("\nGrid search FID...")

try:
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance
    HAS_FID = True
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "pytorch-fid", "--break-system-packages", "-q"])
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance
    HAS_FID = True

import scipy

'''
N_FID       = 1000   # nombre d'images générées par config (augmenter pour plus de précision)
N_REAL      = 1000   # nombre d'images réelles
FID_BATCH   = 40

CFG_VALUES  = [3.0, 5.0]
TOPK_VALUES = [10, 20, 30]

# ── Inception model ────────────────────────────────────────────────────────────
inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(DEVICE).eval()

@torch.no_grad()
def get_inception_features(imgs_01):
    """imgs_01 : (N, 3, H, W) en [0,1] → features (N, 2048)"""
    imgs_resize = F.interpolate(imgs_01, size=(299, 299), mode='bilinear', align_corners=False)
    feats = []
    for i in range(0, len(imgs_resize), FID_BATCH):
        batch = imgs_resize[i:i+FID_BATCH].to(DEVICE)
        feat  = inception(batch)[0].squeeze(-1).squeeze(-1)
        feats.append(feat.cpu())
    return torch.cat(feats, dim=0).numpy()

# ── Features réelles (depuis le dataset) ──────────────────────────────────────
print("  Calcul des features réelles...")
real_imgs = []
for i in range(N_REAL):
    item = raw[i]
    img  = item["img"]
    if img.mode != "RGB":
        img = img.convert("RGB")
    real_imgs.append(transform(img))
real_imgs_t = torch.stack(real_imgs)
real_imgs_t = torch.clamp((real_imgs_t + 1.0) / 2.0, 0, 1)
real_feats  = get_inception_features(real_imgs_t)
mu_real     = np.mean(real_feats, axis=0)
sigma_real  = np.cov(real_feats, rowvar=False)

# ── Grid search ────────────────────────────────────────────────────────────────
results = []
for cfg_val in CFG_VALUES:
    for topk_val in TOPK_VALUES:
        gen_imgs = []
        for start in range(0, N_FID, FID_BATCH):
            bs      = min(FID_BATCH, N_FID - start)
            lbl_rnd = torch.randint(0, 10, (bs,), device=DEVICE)
            with torch.no_grad():
                imgs_g = model.autoregressive_infer_cfg(
                    vqvae=vqvae, B=bs, label_B=lbl_rnd,
                    infer_patch_nums=PATCH_NUMS,
                    cfg=cfg_val, top_k=topk_val, top_p=1.0,
                )
            gen_imgs.append(torch.clamp((imgs_g + 1.0) / 2.0, 0, 1).cpu())

        gen_imgs_t = torch.cat(gen_imgs, dim=0)
        gen_feats  = get_inception_features(gen_imgs_t)
        mu_gen     = np.mean(gen_feats, axis=0)
        sigma_gen  = np.cov(gen_feats, rowvar=False)
        fid        = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        results.append((fid, cfg_val, topk_val))
        print(f"    cfg={cfg_val:.1f}  top_k={topk_val:>3}  FID={fid:.2f}")

results.sort(key=lambda x: x[0])
best_fid, best_cfg, best_topk = results[0]
print(f"\n  Meilleur FID={best_fid:.2f}  →  cfg={best_cfg}  top_k={best_topk}")
'''
# ── Visualisation : 4 samples générés par classe ──────────────────────────────
print("\nGénération de 4 samples par classe...")

N_PER_CLASS = 4
N_CLASSES   = 10

fig, axes = plt.subplots(N_PER_CLASS + 1, N_CLASSES,
                          figsize=(2.2 * N_CLASSES, 2.2 * (N_PER_CLASS + 1)))
fig.patch.set_facecolor('white')
fig.suptitle(f"Generated samples — 4 per class",
             fontsize=13, fontweight='bold', color='#1a1a1a', y=0.99)

# Ligne 0 : labels des classes
for c in range(N_CLASSES):
    axes[0, c].axis('off')
    axes[0, c].text(0.5, 0.5, CIFAR_CLASSES[c],
                    ha='center', va='center', fontsize=10,
                    fontweight='bold', color='#1a1a1a',
                    transform=axes[0, c].transAxes)

for c in range(N_CLASSES):
    label_gen = torch.full((N_PER_CLASS,), fill_value=c, device=DEVICE)

    with torch.no_grad():
        imgs_gen = model.autoregressive_infer_cfg(
            vqvae=vqvae,
            B=N_PER_CLASS,
            label_B=label_gen,
            infer_patch_nums=PATCH_NUMS,
            cfg=3.0,
            top_k=30,
            top_p=1.0,
        )
    imgs_gen = torch.clamp((imgs_gen + 1.0) / 2.0, 0, 1)

    for k in range(N_PER_CLASS):
        img_np = imgs_gen[k].cpu().permute(1, 2, 0).numpy()
        axes[k + 1, c].imshow(img_np, interpolation='nearest')
        axes[k + 1, c].axis('off')
        for spine in axes[k + 1, c].spines.values():
            spine.set_visible(False)

plt.tight_layout(pad=0.5)
out_path = os.path.join(OUT_DIR, "class_samples_grid.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ── Test : génération extrapolée à 32×32 ──────────────────────────────────────
# On étend patch_nums jusqu'à 32 à l'inférence uniquement.
# Toutes les dimensions Matryoshka sont actives à la dernière scale.
print("\n===== Test génération extrapolée 32×32 =====")

PATCH_NUMS_32 = PATCH_NUMS + [32]   # [1,2,4,6,8,10,13,16,32]
N_GEN_32      = 8
CFG_32        = 3.0
TOPK_32       = 30

label_gen_32 = torch.randint(0, 10, (N_GEN_32,), device=DEVICE)

@torch.no_grad()
def autoregressive_infer_extrap(model, vqvae, label_B, patch_nums_ext, B, device,
                                 cfg=3.0, top_k=30, top_p=1.0):
    """
    Génération autorégressive avec patch_nums_ext pouvant dépasser
    les patch_nums d'entraînement. Les dimensions Matryoshka sont
    toutes actives à la dernière scale (active_dim = C).
    """
    vae_emb = F.normalize(vqvae.quantize.embedding.weight, p=2, dim=-1)

    sos = cond_BD = model.class_emb(
        torch.cat((label_B, torch.full_like(label_B, fill_value=model.num_classes)), dim=0)
    )
    class_token = sos.unsqueeze(1)

    # Scale 0 : tokens de départ
    lvl_pos_0       = model.get_pos_embed(patch_nums=patch_nums_ext, si=0)
    scale_tokens_0  = (
        sos.unsqueeze(1).expand(2*B, model.first_l, -1)
        + model.pos_start.expand(2*B, model.first_l, -1)
        + lvl_pos_0
    )
    dim_per_scale = model.C // len(model.patch_nums)   # basé sur l'entraînement
    mask_0 = torch.zeros_like(scale_tokens_0)
    mask_0[:, :, :dim_per_scale] = 1.0
    scale_tokens_0 = scale_tokens_0 * mask_0
    next_token_map = torch.cat([class_token, scale_tokens_0], dim=1)

    cond_BD_or_gss = model.shared_ada_lin(cond_BD)
    all_indices    = []
    idx_Bl         = None

    for si, pn in enumerate(patch_nums_ext):
        ratio = si / max(len(patch_nums_ext) - 1, 1)

        x = next_token_map
        for blk in model.blocks:
            x = blk(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)

        # Si pn dépasse les patch_nums entraînés, on utilise le dernier (toutes dims actives)
        infer_pn_clamped = pn if pn in model.patch_nums else model.patch_nums[-1]
        logits_BlV = model.get_logits(x[:, 1:], cond_BD, infer_pn=infer_pn_clamped)
        t          = cfg * ratio
        logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

        idx_Bl = sample_with_top_k_top_p_(
            logits_BlV, rng=None, top_k=top_k, top_p=top_p, num_samples=1
        )[:, :, 0]
        all_indices.append(idx_Bl)

        if si != len(patch_nums_ext) - 1:
            next_hw    = patch_nums_ext[si + 1]
            curr_hw    = pn
            curr_quant = vae_emb[idx_Bl].reshape(B, curr_hw, curr_hw, vae_emb.shape[-1]).permute(0,3,1,2)
            next_quant = F.interpolate(curr_quant, size=(next_hw, next_hw), mode='bicubic', align_corners=False)
            next_quant = next_quant.reshape(B, curr_quant.shape[1], -1).permute(0, 2, 1)

            scale_tokens_next = model.word_embed(next_quant)
            scale_tokens_next = model.in_norm(scale_tokens_next)
            scale_tokens_next = scale_tokens_next + model.get_pos_embed(patch_nums=patch_nums_ext, si=si+1)

            # Matryoshka : toutes les dims actives à la dernière scale entraînée,
            # et au-delà on garde C entier.
            n_trained_scales = len(model.patch_nums)
            active_scale_idx = min(si + 1, n_trained_scales - 1)
            active_dim       = (active_scale_idx + 1) * dim_per_scale
            active_dim       = min(active_dim, model.C)

            mask_next = torch.zeros_like(scale_tokens_next)
            mask_next[:, :, :active_dim] = 1.0
            scale_tokens_next = scale_tokens_next * mask_next
            scale_tokens_next = scale_tokens_next.repeat(2, 1, 1)

            next_token_map = torch.cat([class_token, scale_tokens_next], dim=1)

    # Décodage de la dernière scale
    last_pn  = patch_nums_ext[-1]
    img_out  = vqvae.decode_code(idx_Bl, shape=(B, vqvae.Cvae, last_pn, last_pn))
    img_out  = torch.clamp((img_out + 1.0) / 2.0, 0, 1)
    return img_out, torch.cat(all_indices, dim=1)


with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    imgs_32, all_idx_32 = autoregressive_infer_extrap(
        model=model, vqvae=vqvae,
        label_B=label_gen_32,
        patch_nums_ext=PATCH_NUMS_32,
        B=N_GEN_32, device=DEVICE,
        cfg=CFG_32, top_k=TOPK_32,
    )

# ── Visualisation : une ligne par sample, une colonne par scale ───────────────
n_scales_32 = len(PATCH_NUMS_32)
fig, axes = plt.subplots(N_GEN_32, n_scales_32,
                          figsize=(2.0 * n_scales_32, 2.0 * N_GEN_32))
fig.patch.set_facecolor('white')
fig.suptitle("Extrapolated generation — scales 1×1 → 32×32  (trained up to 16×16)",
             fontsize=11, fontweight='bold', color='#1a1a1a', y=1.01)

# En-têtes colonnes
for si, pn in enumerate(PATCH_NUMS_32):
    axes[0, si].set_title(f"{pn}×{pn}", fontsize=8, fontweight='bold',
                           color='#E24B4A' if pn == 32 else '#1a1a1a', pad=3)

cur = 0
for si, pn in enumerate(PATCH_NUMS_32):
    seq_len = pn ** 2
    idx_s   = all_idx_32[:, cur:cur + seq_len]   # (B, pn²)

    with torch.no_grad():
        imgs_s = decode_indices(vqvae, idx_s, pn, N_GEN_32)   # (B, 3, pn, pn)

    for b in range(N_GEN_32):
        img_np = F.interpolate(imgs_s[b:b+1], size=(64, 64), mode='nearest')[0]
        img_np = img_np.float().cpu().permute(1, 2, 0).numpy()
        axes[b, si].imshow(img_np, interpolation='nearest')
        axes[b, si].axis('off')
        for spine in axes[b, si].spines.values():
            spine.set_visible(pn == 32)
            spine.set_color('#E24B4A')
            spine.set_linewidth(1.5)
        if si == 0:
            axes[b, si].set_ylabel(CIFAR_CLASSES[label_gen_32[b].item()],
                                    fontsize=7, fontweight='bold', color='#1a1a1a',
                                    rotation=0, labelpad=36, va='center')

    cur += seq_len

plt.tight_layout(pad=0.4)
out_path_32 = os.path.join(OUT_DIR, "extrap_32x32_generation.png")
plt.savefig(out_path_32, dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"  Sauvegardé : {out_path_32}")

# ── Stats : diversité des tokens à 32×32 ──────────────────────────────────────
last_seq_32  = 32 ** 2
last_idx_32  = all_idx_32[:, -last_seq_32:]
n_unique_32  = torch.unique(last_idx_32).numel()
entropy_32   = torch.distributions.Categorical(
    probs=torch.bincount(last_idx_32.flatten(), minlength=1024).float() /
          last_idx_32.numel()
).entropy().item()

print(f"  32×32  tokens_uniques={n_unique_32}/1024  "
      f"entropie={entropy_32:.3f}  (max={torch.log(torch.tensor(1024.)):.3f})")

# Comparer avec 16×16 pour référence
last_seq_16  = 16 ** 2
last_idx_16  = all_idx_32[:, -(last_seq_32 + last_seq_16):-last_seq_32]
n_unique_16  = torch.unique(last_idx_16).numel()
entropy_16   = torch.distributions.Categorical(
    probs=torch.bincount(last_idx_16.flatten(), minlength=1024).float() /
          last_idx_16.numel()
).entropy().item()

print(f"  16×16  tokens_uniques={n_unique_16}/1024  "
      f"entropie={entropy_16:.3f}  (max={torch.log(torch.tensor(1024.)):.3f})")
print(f"\n  → ratio entropie 32/16 : {entropy_32/entropy_16:.3f}  "
      f"(proche de 1 = bonne extrapolation)")

print(f"  Sauvegardé : {out_path}")