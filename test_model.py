import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.models.var import VAR
from src.utils import CIFARDataset
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

patch_nums = [1, 2, 4, 8, 16]
patch_size = 2
num_channels = 3
embed_dim = num_channels * (patch_size ** 2)
num_classes = 10
last_pn = patch_nums[-1]
target_res = last_pn * patch_size

model = VAR(
    num_classes=num_classes,
    num_channels=num_channels,
    patch_nums=patch_nums,
    patch_size=patch_size,
    pixel_dim=embed_dim,
    man_dim=64,
    embed_dim=768
).to(device)

checkpoint_path = "checkpoints/model_epoch_2.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded: {checkpoint_path}")

dataset = CIFARDataset(patch_nums=patch_nums, patch_size=patch_size)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

scale_seq, label_b, original_image = next(iter(dataloader))
scale_seq = scale_seq.to(device)
label_b = label_b.to(device)

model.eval()
with torch.no_grad():
    # 1. Teacher Forcing Complet
    first_l = patch_nums[0] ** 2
    x_input_tf = scale_seq[:, first_l:]
    scale_seq_pred_tf, _ = model(label_b, x_input_tf)

    # 2. Full Autoregressive (Part de zéro)
    ar_full = model.autoregressive_infer_cfg(B=1, label_B=label_b, cfg=0.0)

    # 3. Semi-Autoregressive (On donne le premier scale)
    # On récupère juste le premier scale de la séquence réelle
    x_first_scale = scale_seq[:, :first_l] 
    ar_semi = model.autoregressive_infer_cfg(B=1, label_B=label_b, cfg=0.0, start_scale_data=x_first_scale)

# Visualisation
fig, axes = plt.subplots(3, len(patch_nums) + 1, figsize=(3 * (len(patch_nums) + 1), 9))

gt_accum = torch.zeros(1, num_channels, target_res, target_res, device=device)
tf_accum = torch.zeros(1, num_channels, target_res, target_res, device=device)

cur_idx = 0
for i, pn in enumerate(patch_nums):
    num_patches = pn * pn
    
    # GT Accumulation
    gt_p = scale_seq[:, cur_idx:cur_idx+num_patches].reshape(1, pn, pn, num_channels, patch_size, patch_size)
    gt_pix = gt_p.permute(0, 3, 1, 4, 2, 5).reshape(1, num_channels, pn * patch_size, pn * patch_size)
    gt_accum += F.interpolate(gt_pix, size=(target_res, target_res), mode='bicubic')
    
    # TF Accumulation
    tf_p = scale_seq_pred_tf[:, cur_idx:cur_idx+num_patches].reshape(1, pn, pn, num_channels, patch_size, patch_size)
    tf_pix = tf_p.permute(0, 3, 1, 4, 2, 5).reshape(1, num_channels, pn * patch_size, pn * patch_size)
    tf_accum += F.interpolate(tf_pix, size=(target_res, target_res), mode='bicubic')
    
    axes[0, i].imshow(np.clip((gt_accum[0].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0, 0, 1))
    axes[0, i].set_title(f"GT {pn}x{pn}")
    axes[1, i].imshow(np.clip((tf_accum[0].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0, 0, 1))
    axes[1, i].set_title(f"TF {pn}x{pn}")
    axes[2, i].axis('off') # On laisse vide pour les colonnes intermédiaires de la ligne 3
    
    for r in range(2): axes[r, i].axis('off')
    cur_idx += num_patches

# Comparaison finale
orig_np = np.clip((original_image[0].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0, 0, 1)
axes[0, -1].imshow(orig_np)
axes[0, -1].set_title("Original GT")

axes[1, -1].imshow(np.clip(ar_full[0].permute(1, 2, 0).cpu().numpy(), 0, 1))
axes[1, -1].set_title("Full AR (from 0)")

axes[2, -1].imshow(np.clip(ar_semi[0].permute(1, 2, 0).cpu().numpy(), 0, 1))
axes[2, -1].set_title("Semi AR (from Scale 1)")

for r in range(3): axes[r, -1].axis('off')

plt.tight_layout()
plt.savefig("test_visualisation_v2.png", bbox_inches='tight', dpi=150)
plt.close(fig)
print("Saved to test_visualisation_v2.png")