from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import torch.nn.functional as F
import torch

class TinyImageDataset(Dataset):
    def __init__(self, data_path='zh-plus/tiny-imagenet', patch_nums=[1, 2, 3, 4], patch_size=16):
        self.dataset = load_dataset(data_path, split='train', cache_dir='src/data')
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        last_pn = self.patch_nums[-1]
        ps = self.patch_size
        
        img_raw = self.dataset['image'][index]
        if img_raw.mode != 'RGB':
            img_raw = img_raw.convert('RGB')
        
        img_np = np.array(img_raw).astype(np.float32)
        img_tensor = torch.from_numpy(img_np)
        img_normalized = (img_tensor / 127.5) - 1.0  # (64, 64, 3)
        
        img_CHW = img_normalized.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 64, 64)
        
        label_b = self.dataset['label'][index]
        x_BLCv_wo_first_l = []
        
        img_residual = img_CHW.clone()
        
        for pn in self.patch_nums:
            target_size = pn * ps 
            residual_pixels = F.interpolate(img_residual, size=(target_size, target_size), 
                                           mode='bicubic', align_corners=False)
            
            # (1, 3, target_size, target_size) -> (pn, pn, 3*ps*ps)
            residual_patches = residual_pixels.squeeze(0)  # (3, target_size, target_size)
            residual_patches = residual_patches.reshape(3, pn, ps, pn, ps)  # (3, pn, ps, pn, ps)
            residual_patches = residual_patches.permute(1, 3, 0, 2, 4)  # (pn, pn, 3, ps, ps)
            residual_patches = residual_patches.reshape(pn, pn, 3 * ps * ps)  # (pn, pn, 768)
            
            x_BLCv_wo_first_l.append(residual_patches.reshape(pn * pn, -1))
            
            reconstructed_pixels = F.interpolate(residual_pixels, size=(last_pn * ps, last_pn * ps), 
                                                 mode='bicubic', align_corners=False)
            
            img_residual = img_residual - reconstructed_pixels
        
        x_BLCv_wo_first_l = torch.cat(x_BLCv_wo_first_l, dim=0)
        
        return x_BLCv_wo_first_l, label_b

