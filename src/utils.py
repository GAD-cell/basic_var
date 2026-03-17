from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import torch.nn.functional as F
import torch

class CIFARDataset(Dataset):
    def __init__(self, patch_nums=[1, 2, 4, 8, 16], patch_size=2):
        self.dataset = load_dataset('cifar10', split='train', cache_dir='src/data')
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        last_pn = self.patch_nums[-1]
        ps = self.patch_size
        target_res = last_pn * ps
        
        img_raw = self.dataset[index]['img']
        if img_raw.mode != 'RGB':
            img_raw = img_raw.convert('RGB')
        
        img_np = np.array(img_raw).astype(np.float32)
        img_tensor = torch.from_numpy(img_np)
        img_normalized = (img_tensor / 127.5) - 1.0
        
        img_CHW = img_normalized.permute(2, 0, 1).unsqueeze(0)
        
        if img_CHW.shape[-1] != target_res:
            img_CHW = F.interpolate(img_CHW, size=(target_res, target_res), mode='bicubic', align_corners=False)
        
        label_b = self.dataset[index]['label']
        x_BLCv_wo_first_l = []
        
        for pn in self.patch_nums:
            target_size = pn * ps 
            scaled_pixels = F.interpolate(img_CHW, size=(target_size, target_size), mode='bicubic', align_corners=False)
            
            patches = scaled_pixels.squeeze(0)
            patches = patches.reshape(3, pn, ps, pn, ps)
            patches = patches.permute(1, 3, 0, 2, 4)
            patches = patches.reshape(pn, pn, 3 * ps * ps)
            
            x_BLCv_wo_first_l.append(patches.reshape(pn * pn, -1))
            
        x_BLCv_wo_first_l = torch.cat(x_BLCv_wo_first_l, dim=0)
        
        return x_BLCv_wo_first_l, label_b, img_CHW.squeeze(0)