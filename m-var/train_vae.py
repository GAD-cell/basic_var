import os
import sys
from pathlib import Path
import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.vq_llama import VQ_8

class CIFAR10VAEDataset(Dataset):
    def __init__(self, target_size=32, split='train'):
        self.dataset = load_dataset('cifar10', split=split, cache_dir='src/data')
        self.target_size = target_size
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img_raw = self.dataset[index]['img']
        if img_raw.mode != 'RGB':
            img_raw = img_raw.convert('RGB')
            
        img_np = np.array(img_raw).astype(np.float32)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        
        if img_tensor.shape[-1] != self.target_size:
            img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(self.target_size, self.target_size), mode='bicubic', align_corners=False).squeeze(0)
            
        img_normalized = (img_tensor / 127.5) - 1.0
        return img_normalized

def evaluate_rfid(model, dataloader, device):
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    model.eval()
    
    with torch.no_grad():
        for real_imgs in dataloader:
            real_imgs = real_imgs.to(device)
            real_imgs_uint8 = ((real_imgs.clamp(-1, 1) + 1) / 2)
            fid.update(real_imgs_uint8, real=True)
            
            reconstructed, _ = model(real_imgs)
            reconstructed_uint8 = (reconstructed.clamp(-1, 1) + 1) / 2
            fid.update(reconstructed_uint8, real=False)
            
    score = fid.compute().item()
    model.train()
    return score

def train_vqvae():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    os.makedirs("checkpoints_vae", exist_ok=True)
    os.makedirs("outputs_vae", exist_ok=True)

    vqvae = VQ_8().to(device)
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    
    optimizer = optim.AdamW(vqvae.parameters(), lr=2e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    
    train_dataset = CIFAR10VAEDataset(target_size=32, split='train')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    test_dataset = CIFAR10VAEDataset(target_size=32, split='test')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    num_epochs = 200

    for epoch in range(num_epochs):
        vqvae.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for imgs in pbar:
            imgs = imgs.to(device)
            
            recon_imgs, diff = vqvae(imgs)
            vq_loss, commit_loss, entropy_loss, codebook_usage = diff
            
            recon_loss = F.l1_loss(recon_imgs, imgs) + F.mse_loss(recon_imgs, imgs)
            p_loss = perceptual_loss_fn(recon_imgs, imgs).mean()
            
            total_loss = recon_loss + 0.4 * p_loss + vq_loss + 1.2 * commit_loss + entropy_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}", 
                recon=f"{recon_loss.item():.4f}",
                lpips=f"{p_loss.item():.4f}",
                usage=f"{codebook_usage:.2%}"
            )

        print(f"\nComputing rFID at epoch {epoch+1}...")
        rfid_score = evaluate_rfid(vqvae, test_loader, device)
        print(f"--- Epoch {epoch+1} | rFID: {rfid_score:.4f} ---")

        if (epoch + 1) % 5 == 0:
            vqvae.eval()
            with torch.no_grad():
                sample_imgs = imgs[:8]
                recon_sample, _ = vqvae(sample_imgs)
                
                comparison = torch.cat([sample_imgs, recon_sample], dim=0)
                save_image(comparison, f"outputs_vae/recon_e{epoch+1}.png", nrow=8, normalize=True, value_range=(-1, 1))
                
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': vqvae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rfid': rfid_score
            }, f"checkpoints_vae/vqvae_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_vqvae()