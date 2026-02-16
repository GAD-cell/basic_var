import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from src.models.var import VAR
from src.utils import TinyImageDataset

def save_generated_image(model, epoch, step, device, save_dir, num_samples=4):
    model.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
        if num_samples == 1: axes = [axes]
        
        for i in range(num_samples):
            generated_img = model.autoregressive_infer_cfg(
                B=1, 
                label_B=torch.tensor([i % 200]).to(device), 
                cfg=1.5
            )
            img_np = generated_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img_np = np.clip((img_np + 1.0) / 2.0, 0, 1)
            axes[i].imshow(img_np)
            axes[i].axis('off')
        
        plt.savefig(os.path.join(save_dir, f"gen_e{epoch}_s{step}.png"))
        plt.close()
    model.train()

def train():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    patch_nums = [1, 2, 3, 4]
    patch_size = 16
    num_channels = 3
    embed_dim = num_channels * (patch_size ** 2)
    num_classes = 200
    
    batch_size = 32 
    num_epochs = 100
    gen_interval = 100
    lambda_root = 0.5
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Chargement DINOv2
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
    dino.eval()
    for param in dino.parameters():
        param.requires_grad = False
    
    dino_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    mse_criterion = torch.nn.MSELoss()

    global_step = 0
    model.train()

    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for scale_seq, label_b, original_image in progress_bar:
            scale_seq = scale_seq.to(device)
            label_b = label_b.to(device)
            original_image = original_image.to(device)

            with torch.no_grad():
                img_224 = F.interpolate((original_image + 1.0) / 2.0, size=(224, 224), mode='bicubic')
                target_features = dino(dino_norm(img_224)) # (B, 768)
                target_features = F.normalize(target_features, p=2, dim=-1)

            x_input = scale_seq[:, patch_nums[0]**2:]
            scale_seq_pred, latent_rep = model(label_b, x_input)
            

            root_latent = latent_rep[:, 0, :] # (B, 768)
            root_latent = F.normalize(root_latent, p=2, dim=-1)
            
            loss_mse = mse_criterion(scale_seq_pred, scale_seq)
            loss_root = 1 - F.cosine_similarity(root_latent, target_features, dim=-1).mean()
            
            total_loss = loss_mse + lambda_root * loss_root
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            global_step += 1
            progress_bar.set_postfix(mse=f"{loss_mse.item():.4f}", root=f"{loss_root.item():.4f}")

            if global_step % gen_interval == 0:
                save_generated_image(model, epoch + 1, global_step, device, "outputs")

        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()