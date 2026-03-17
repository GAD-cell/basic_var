import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import ot
import gc
from src.models.var import VAR
from src.utils import CIFARDataset

def get_last_scale_img(scale_seq, patch_nums, patch_size, num_channels):
    B = scale_seq.shape[0]
    ps = patch_size
    last_pn = patch_nums[-1]
    num_patches = last_pn * last_pn
    patches = scale_seq[:, -num_patches:, :].reshape(B, last_pn, last_pn, num_channels, ps, ps)
    return patches.permute(0, 3, 1, 4, 2, 5).reshape(B, num_channels, last_pn * ps, last_pn * ps)

def get_all_scales_images(scale_seq, patch_nums, patch_size, num_channels):
    B = scale_seq.shape[0]
    ps = patch_size
    images = []
    cur_idx = 0
    for pn in patch_nums:
        num_patches = pn * pn
        patches = scale_seq[:, cur_idx:cur_idx+num_patches, :].reshape(B, pn, pn, num_channels, ps, ps)
        pixels = patches.permute(0, 3, 1, 4, 2, 5).reshape(B, num_channels, pn * ps, pn * ps)
        images.append(pixels)
        cur_idx += num_patches
    return images

def save_generated_image(model, epoch, step, device, save_dir, scale_seq_pred, original_image, labels, patch_nums, patch_size, num_channels):
    model.eval()
    with torch.no_grad():
        B_avail = min(4, scale_seq_pred.shape[0])
        recon_tf = get_last_scale_img(scale_seq_pred[:B_avail], patch_nums, patch_size, num_channels)
        fig, axes = plt.subplots(3, B_avail, figsize=(4 * B_avail, 12))
        if B_avail == 1: axes = axes[:, np.newaxis]
        for i in range(B_avail):
            axes[0, i].imshow(np.clip((original_image[i].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0, 0, 1))
            axes[1, i].imshow(np.clip((recon_tf[i].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0, 0, 1))
            gen = model.autoregressive_infer_cfg(B=1, label_B=labels[i].unsqueeze(0), cfg=0.0)
            axes[2, i].imshow(np.clip(gen[0].permute(1, 2, 0).cpu().numpy(), 0, 1))
            for r in range(3): axes[r, i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"gen_e{epoch}_s{step}.png"))
        plt.close(fig)
    model.train()

def train(checkpoint_path=None):
    torch.cuda.empty_cache()
    gc.collect()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading DINOv2 (VITS14)...")
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    dino.eval()
    for param in dino.parameters(): param.requires_grad = False
    dino_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    patch_nums = [1, 2, 4, 8, 16]
    patch_size, num_channels, num_classes = 2, 3, 10
    embed_dim = num_channels * (patch_size ** 2)
    
    batch_size = 32 # Augmenté car A40 est large
    num_epochs = 100
    stage_split_epoch = 40
    
    model = VAR(num_classes=num_classes, num_channels=num_channels, patch_nums=patch_nums, 
                patch_size=patch_size, pixel_dim=embed_dim, man_dim=64, embed_dim=768).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    dataset = CIFARDataset(patch_nums=patch_nums, patch_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    start_epoch, global_step = 0, 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch, global_step = ckpt['epoch'], ckpt['global_step']

    for epoch in range(start_epoch, num_epochs):
        is_stage_2 = epoch >= stage_split_epoch
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Stage {2 if is_stage_2 else 1}]")
        
        for scale_seq, label_b, original_image in pbar:
            scale_seq, label_b, original_image = scale_seq.to(device), label_b.to(device), original_image.to(device)

            with torch.no_grad():
                img_target_224 = F.interpolate((original_image + 1.0) / 2.0, size=(224, 224), mode='bicubic')
                feat_target = F.normalize(dino(dino_norm(img_target_224)), p=2, dim=-1)

            first_l = patch_nums[0] ** 2
            x_input = scale_seq[:, first_l:].clone()
            if torch.rand(1).item() < 0.5:
                x_input += torch.randn_like(x_input) * 0.1
            
            pred, _ = model(label_b, x_input)
            
            loss_l1 = F.l1_loss(pred, scale_seq)
            
            loss_perc = 0.0
            if is_stage_2:
                all_imgs = get_all_scales_images(pred, patch_nums, patch_size, num_channels)
                for img in all_imgs:
                    img_224 = F.interpolate((img + 1.0) / 2.0, size=(224, 224), mode='bicubic')
                    feat_pred = F.normalize(dino(dino_norm(img_224)), p=2, dim=-1)
                    loss_perc += (1.0 - F.cosine_similarity(feat_pred, feat_target, dim=-1).mean())
                loss_perc /= len(patch_nums)
            else:
                last_img = get_last_scale_img(pred, patch_nums, patch_size, num_channels)
                img_224 = F.interpolate((last_img + 1.0) / 2.0, size=(224, 224), mode='bicubic')
                feat_pred = F.normalize(dino(dino_norm(img_224)), p=2, dim=-1)
                loss_perc = (1.0 - F.cosine_similarity(feat_pred, feat_target, dim=-1).mean())

            # SWD sur le premier scale (Global structure)
            p_first = pred[:, :first_l, :].reshape(pred.shape[0], -1)
            t_first = scale_seq[:, :first_l, :].reshape(pred.shape[0], -1)
            loss_swd = ot.sliced_wasserstein_distance(p_first, t_first, n_projections=128)
            
            total_loss = loss_l1 + (0.5 * loss_perc) + (0.5 * loss_swd)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            global_step += 1
            pbar.set_postfix(l1=f"{loss_l1.item():.3f}", perc=f"{loss_perc.item():.3f}", swd=f"{loss_swd.item():.3f}")

            if global_step % 200 == 0:
                save_generated_image(model, epoch+1, global_step, device, "outputs", pred, original_image, label_b, patch_nums, patch_size, num_channels)

        torch.save({'epoch': epoch+1, 'global_step': global_step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f"checkpoints/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()