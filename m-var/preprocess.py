import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm
import os
from huggingface_hub import HfApi, login
from models.vq_llama import VQ_8

class CIFARHuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        img = self.hf_dataset[idx]['img']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_tensor = self.transform(img)
        label = self.hf_dataset[idx]['label']
        return img_tensor, label

@torch.no_grad()
def preprocess_and_upload():
    # --- Configuration ---
    device = "cuda"
    v_patch_nums = [1, 2, 4, 6, 8, 10, 13, 16]
    output_file = "cifar10_flexvar_precomputed.pt"
    repo_id = "GAD-cell/cifar10-flexvar-precomputed" # REMPLACE PAR TON USERNAME
    
    # 3. Setup Modèle et Données
    vqvae = VQ_8().to(device)
    vqvae.load_state_dict(torch.load("checkpoints_vae/vqvae_epoch_10.pth")["model_state_dict"])
    vqvae.eval()
    
    vae_embedding = F.normalize(vqvae.quantize.embedding.weight, p=2, dim=-1).to(device)
    
    raw_dataset = load_dataset('cifar10', split='train')
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = CIFARHuggingFaceDataset(raw_dataset, transform)
    dataloader = DataLoader(dataset, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
    
    precomputed_data = []
    L = sum(pn ** 2 for pn in v_patch_nums)
    first_l = v_patch_nums[0] ** 2

    # 4. Boucle de Preprocessing
    print(f"Démarrage du preprocessing sur {device}...")
    for imgs, labels in tqdm(dataloader):
        B = imgs.size(0)
        imgs = imgs.to(device)
        
        h = vqvae.encode_conti(imgs)
        
        batch_indices = []
        batch_inputs = []
        
        for j, curr_hw in enumerate(v_patch_nums):
            _h = F.interpolate(h, size=(curr_hw, curr_hw), mode='area')
            _, _, log = vqvae.quantize(_h)
            
            idx = log[-1].view(B, -1)
            batch_indices.append(idx.cpu())
            
            if j < len(v_patch_nums) - 1:
                next_hw = v_patch_nums[j + 1]
                curr_quant = vae_embedding[idx]
                curr_quant = curr_quant.reshape(B, curr_hw, curr_hw, -1).permute(0, 3, 1, 2)
                
                next_quant = F.interpolate(curr_quant, size=(next_hw, next_hw), mode='bicubic')
                next_quant = next_quant.reshape(B, -1, next_hw * next_hw).permute(0, 2, 1)
                
                batch_inputs.append(next_quant.cpu())
                
        target_indices = torch.cat(batch_indices, dim=1)
        x_input_quants = torch.cat(batch_inputs, dim=1)
        
        for b in range(B):
            precomputed_data.append({
                'x_input_quants': x_input_quants[b],
                'target_indices': target_indices[b],
                'label': labels[b].item()
            })
            
    # 5. Sauvegarde et Upload
    print(f"Sauvegarde locale de {output_file}...")
    torch.save(precomputed_data, output_file)


if __name__ == "__main__":
    preprocess_and_upload()