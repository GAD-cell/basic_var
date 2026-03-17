import os
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb
from models.flexvar import FlexVAR
from models.vq_llama import VQ_8
import torchvision
from torch.amp import autocast
from utils import CIFARPrecomputedDataset, CIFARDataset

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if os.path.exists("checkpoints_vae/flexvar_mat.pth"):
        checkpoint = torch.load("checkpoints_vae/flexvar_mat.pth", map_location=device)
    else:
        checkpoint = {}
    #checkpoint = {}
    wandb.init(
        project="FlexVAR-CIFAR",
        config={
            "learning_rate": 1e-4,
            "architecture": "FlexVAR",
            "dataset": "CIFAR-10",
            "epochs": 200,
            "batch_size": 64,
            "patch_nums": [1, 2, 4, 6, 8, 10, 13, 16]
        }
    )

    SAVE_EVERY_X_STEPS = 200
    global_step = checkpoint.get("step", 0)

    vqvae = VQ_8().to(device)
    vqvae.load_state_dict(torch.load("checkpoints_vae/vqvae_epoch_10.pth", map_location=device)["model_state_dict"])
    vqvae.eval()
    for p in vqvae.parameters(): 
        p.requires_grad = False

    patch_nums = wandb.config.patch_nums
    model = FlexVAR(
        vae_local=vqvae,
        num_classes=10,
        depth=12,
        embed_dim=512,
        num_heads=8,
        patch_nums=patch_nums,
        attn_l2_norm=True,
    ).to(device)
    
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=wandb.config.learning_rate, 
        weight_decay=0.2, 
        betas=(0.9, 0.95)
    )
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
     
    dataset = CIFARPrecomputedDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=wandb.config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=wandb.config.learning_rate,
        steps_per_epoch=len(dataloader),
        epochs=wandb.config.epochs,
        pct_start=0.1 
    )
    
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    os.makedirs("outputs", exist_ok=True)
    scale_weights = {1: 1.0, 2: 0.1, 4: 0.05, 6: 0.2, 8: 1.0, 10: 0.5, 13: 0.5, 16: 1.0}

    for epoch in range(wandb.config.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for x_input_quants, target_indices, label_b in pbar:
            x_input_quants = x_input_quants.to(device)
            target_indices = target_indices.to(device)
            label_b = label_b.to(device)

            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(label_b, x_input_quants, patch_nums)
                if logits.requires_grad:
                    logits.retain_grad()
                
                total_loss = 0.0
                cur_len = 0
                max_pn = patch_nums[-1]

                for pn in patch_nums:
                    seq_len = pn ** 2
                    scale_logits = logits[:, cur_len : cur_len + seq_len, :]
                    scale_targets = target_indices[:, cur_len : cur_len + seq_len]
                    

                    scale_loss = F.cross_entropy(
                            scale_logits.reshape(-1, scale_logits.size(-1)), 
                            scale_targets.reshape(-1),
                        )
                        
                    total_loss += scale_loss * scale_weights.get(pn, 1.0)
                    
                    cur_len += seq_len
                
                loss = total_loss / len(patch_nums)

            loss.backward()

            layer_grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if "word_embed" in name or "lvl_embed" in name:
                        layer_grad_norms[f"grad_layer/input_{name}"] = param.grad.norm(2).item()
                    elif "blocks.0." in name:
                        layer_grad_norms[f"grad_layer/first_block_{name.split('.')[-1]}"] = param.grad.norm(2).item()
                    elif "blocks.6." in name:
                        layer_grad_norms[f"grad_layer/mid_block_{name.split('.')[-1]}"] = param.grad.norm(2).item()
                    elif "blocks.11." in name:
                        layer_grad_norms[f"grad_layer/last_block_{name.split('.')[-1]}"] = param.grad.norm(2).item()
                    elif "head" in name:
                        layer_grad_norms[f"grad_layer/head_{name.split('.')[-1]}"] = param.grad.norm(2).item()
            
            scale_grad_norms = {}
            if logits.grad is not None:
                cur_len = 0
                for pn in patch_nums:
                    seq_len = pn ** 2
                    scale_grad = logits.grad[:, cur_len : cur_len + seq_len, :]
                    scale_grad_norms[f"grad_logits_scale_{pn}"] = scale_grad.norm(2).item()
                    cur_len += seq_len

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()

            log_data = {
                "train_loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm.item()
            }
            log_data.update(scale_grad_norms)
            log_data.update(layer_grad_norms)
            wandb.log(log_data, step=global_step)

            pbar.set_postfix(loss=f"{loss.item():.4f}", norm=f"{grad_norm.item():.2f}")

            if global_step % SAVE_EVERY_X_STEPS == 0:
                save_and_log_wandb(model, vqvae, global_step, device, patch_nums, x_input_quants, target_indices, label_b)
                model.train()

            global_step += 1

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'step': global_step,
        }, "checkpoints_vae/flexvar_latest.pth")

    wandb.finish()

@torch.no_grad()
def save_and_log_wandb(model, vqvae, step, device, patch_nums, x_input_quants, target_indices, label_b):
    model.eval()
    B = min(8, x_input_quants.shape[0])
    labels = torch.arange(B).to(device) % 10
    img_gen = model.autoregressive_infer_cfg(vqvae=vqvae, B=B, label_B=labels, infer_patch_nums=patch_nums, cfg=1.5, top_k=10, top_p=0.90)
    
    x_in, t_ind, lbl = x_input_quants[:B], target_indices[:B], label_b[:B]
    logits = model(lbl, x_in, patch_nums)
    pred_indices = logits.argmax(dim=-1)
    
    last_pn = patch_nums[-1]
    gt_idx = t_ind[:, -last_pn**2:]
    gt_img = vqvae.decode_code(gt_idx, shape=(B, vqvae.Cvae, last_pn, last_pn))
    final_res = gt_img.shape[-2:]
    all_images = [gt_img] 
    cur = 0
    for pn in patch_nums:
        length = pn ** 2
        scale_pred_idx = pred_indices[:, cur:cur+length]
        scale_img = vqvae.decode_code(scale_pred_idx, shape=(B, vqvae.Cvae, pn, pn))
        if scale_img.shape[-2:] != final_res:
            scale_img = F.interpolate(scale_img, size=final_res, mode='bilinear', align_corners=False)
        all_images.append(scale_img)
        cur += length
    grid_gen = torchvision.utils.make_grid(img_gen, nrow=4, normalize=True, value_range=(-1, 1))
    grid_recon = torchvision.utils.make_grid(torch.cat(all_images, dim=0), nrow=B, normalize=True, value_range=(-1, 1))
    wandb.log({"free_generation": wandb.Image(grid_gen), "reconstruction": wandb.Image(grid_recon)})

if __name__ == "__main__":
    train()