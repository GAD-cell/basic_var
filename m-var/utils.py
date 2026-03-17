from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import torch.nn.functional as F
import torch

class CIFARDataset(Dataset):
    def __init__(self, vqvae, v_patch_nums=(1, 2, 4, 8, 16), device='cuda', split='train'):
        
        self.dataset = load_dataset('cifar10', split=split)
        self.vqvae = vqvae.to(device).eval()
        self.v_patch_nums = v_patch_nums
        self.device = device
        
        with torch.no_grad():
            self.vae_embedding = F.normalize(
                vqvae.quantize.embedding.weight, p=2, dim=-1
            ).to(device)  # [V, Cvae]

    def __len__(self):
        return len(self.dataset)

    @torch.no_grad()
    def __getitem__(self, index):
        img_raw = self.dataset[index]['img']
        if img_raw.mode != 'RGB':
            img_raw = img_raw.convert('RGB')

        img_np = np.array(img_raw).astype(np.float32)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if img_tensor.shape[-2] != 32 or img_tensor.shape[-1] != 32:
            img_tensor = F.interpolate(img_tensor, size=(32, 32), mode='bicubic')

        img_normalized = (img_tensor / 127.5) - 1.0

        h = self.vqvae.encode_conti(img_normalized)  # [1, Cvae, H, W]

        all_indices = []
        all_inputs = []

        for i, curr_hw in enumerate(self.v_patch_nums):
            _h = F.interpolate(h, size=(curr_hw, curr_hw), mode='area')
            _, _, log = self.vqvae.quantize(_h)
            idx = log[-1].view(-1)  # [curr_hw²]
            all_indices.append(idx)

            if i < len(self.v_patch_nums) - 1:
                next_hw = self.v_patch_nums[i + 1]

                curr_quant = self.vae_embedding[idx]  # [curr_hw², Cvae]
                curr_quant = curr_quant.reshape(1, curr_hw, curr_hw, -1).permute(0, 3, 1, 2)
                # [1, Cvae, curr_hw, curr_hw]

                next_quant = F.interpolate(curr_quant, size=(next_hw, next_hw), mode='bicubic')
                # [1, Cvae, next_hw, next_hw]

                next_quant = next_quant.reshape(1, -1, next_hw * next_hw).permute(0, 2, 1).squeeze(0)
                # [next_hw², Cvae]

                all_inputs.append(next_quant)

        L = sum(pn ** 2 for pn in self.v_patch_nums)
        first_l = self.v_patch_nums[0] ** 2

        target_indices = torch.cat(all_indices, dim=0)    # [L]
        x_input_quants = torch.cat(all_inputs, dim=0)     # [L - first_l, Cvae]

        assert target_indices.shape[0] == L, f"{target_indices.shape[0]} != {L}"
        assert x_input_quants.shape[0] == L - first_l, f"{x_input_quants.shape[0]} != {L - first_l}"

        label_b = self.dataset[index]['label']

        return x_input_quants.cpu(), target_indices.cpu(), label_b
    
class CIFARPrecomputedDataset(Dataset):
    def __init__(self, pt_path="cifar10_flexvar_precomputed.pt"):
        data = torch.load(pt_path, map_location='cpu')
        self.x_input_quants = torch.stack([d['x_input_quants'] for d in data])
        self.target_indices = torch.stack([d['target_indices'] for d in data])
        self.labels = torch.tensor([d['label'] for d in data])
        
        idx = torch.randperm(len(self.labels))
        self.x_input_quants = self.x_input_quants[idx]
        self.target_indices = self.target_indices[idx]
        self.labels = self.labels[idx]
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        #print(self.x_input_quants.shape, self.target_indices.shape)
        return self.x_input_quants[index], self.target_indices[index], self.labels[index]