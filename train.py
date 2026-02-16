import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.var import VAR
from src.utils import TinyImageDataset

def save_generated_image(model, epoch, step, device, save_dir):
    model.eval()
    with torch.no_grad():
        inference = model.autoregressive_infer_cfg(B=1)
        
        patches = inference[0, -16:, :]
        patches_reshaped = patches.view(16, 3, 16, 16)
        grid = patches_reshaped.view(4, 4, 3, 16, 16)
        final_image = grid.permute(2, 0, 3, 1, 4).contiguous().view(3, 64, 64)
        
        img_np = final_image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np + 1.0) / 2.0
        img_np = np.clip(img_np, 0, 1)
        
        plt.figure(figsize=(4, 4))
        plt.imshow(img_np)
        plt.axis('off')
        plt.title(f"Epoch {epoch} - Step {step}")
        plt.savefig(os.path.join(save_dir, f"gen_e{epoch}_s{step}.png"))
        plt.close()
    model.train()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class DINOLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        print("Loading DINOv2...")
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
        self.dino.eval()
        
        for param in self.dino.parameters():
            param.requires_grad = False
            
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])

    def forward(self, scale_seq_pred, scale_seq, patch_nums):
        """
        scale_seq_pred: (B, Total_Patches, Embed_Dim) - Prédictions (pixels aplatis)
        scale_seq: (B, Total_Patches, Embed_Dim) - Target (pixels aplatis)
        patch_nums: Liste des échelles [1, 2, 3, 4]
        """
        total_loss = 0
        start_idx = 0
        patch_size = 16
        
        for pn in patch_nums:
            num_patches = pn**2
            end_idx = start_idx + num_patches

            pred_chunk = scale_seq_pred[:, start_idx:end_idx, :]
            target_chunk = scale_seq[:, start_idx:end_idx, :]

            pred_imgs = pred_chunk.reshape(-1, 3, patch_size, patch_size)
            target_imgs = target_chunk.reshape(-1, 3, patch_size, patch_size)
            
            pred_imgs = (pred_imgs + 1.0) * 0.5
            target_imgs = (target_imgs + 1.0) * 0.5
            
            pred_large = F.interpolate(pred_imgs, size=(224, 224), mode='bicubic', align_corners=False)
            target_large = F.interpolate(target_imgs, size=(224, 224), mode='bicubic', align_corners=False)
            
            pred_large = self.normalize(pred_large)
            target_large = self.normalize(target_large)
            
            with torch.no_grad():
                target_features = self.dino(target_large)
            
            pred_features = self.dino(pred_large)

            similarity = F.cosine_similarity(pred_features, target_features, dim=-1).mean()
            total_loss += (1 - similarity)
            
            start_idx = end_idx

        return total_loss / len(patch_nums)

def train():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    patch_nums = [1, 2, 3, 4]
    patch_size = 16
    num_channels = 3
    embed_dim = num_channels * (patch_size ** 2)
    num_classes = 200
    
    batch_size = 32
    num_epochs = 100
    save_interval = 5
    gen_interval = 100
    learning_rate = 3e-4
    
    checkpoint_dir = "checkpoints"
    output_dir = "outputs"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    dataset = TinyImageDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VAR(
        num_classes=num_classes,
        num_channels=num_channels,
        patch_nums=patch_nums,
        pixel_dim=embed_dim,
        man_dim=64,
        embed_dim=embed_dim
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    dino_criterion = DINOLoss(device)

    global_step = 0
    model.train()

    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for scale_seq, label_b in progress_bar:
            scale_seq = scale_seq.to(device)
            label_b = label_b.to(device)
            
            x_input = scale_seq[:, patch_nums[0]**2:]
            
            optimizer.zero_grad()
            scale_seq_pred, _ = model(label_b, x_input)
            loss_mse = criterion(scale_seq_pred, scale_seq)
            loss_dino = dino_criterion(scale_seq_pred, scale_seq, patch_nums)
            
            total_loss = loss_mse + 0.1 * loss_dino
            total_loss.backward()
            optimizer.step()
            
            global_step += 1
            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")

            if global_step % gen_interval == 0:
                save_generated_image(model, epoch + 1, global_step, device, output_dir)

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

if __name__ == "__main__":
    train()