import os
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.var import VAR
from src.utils import TinyImageDataset

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
    learning_rate = 3e-4
    save_dir = "checkpoints"

    os.makedirs(save_dir, exist_ok=True)

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

    model.train()

    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for scale_seq, label_b in progress_bar:
            scale_seq = scale_seq.to(device)
            label_b = label_b.to(device)
            
            x_input = scale_seq[:, patch_nums[0]**2:]
            
            optimizer.zero_grad()
            
            scale_seq_pred, _ = model(label_b, x_input)
            
            loss = criterion(scale_seq_pred, scale_seq)
            
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

if __name__ == "__main__":
    train()