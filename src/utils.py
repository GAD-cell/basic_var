from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import torch.nn.functional as F
import torch

class TinyImageDataset(Dataset):
    def __init__(self,data_path='zh-plus/tiny-imagenet', patch_nums=[1,2,3,4],patch_size=16):
        self.dataset = load_dataset(data_path,split='train',cache_dir='src/data')
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        
    def __len__(self,):
        return len(self.dataset)
    
    def __getitem__(self, index):
        last_pn = self.patch_nums[-1]
        ps = self.patch_size
        
        img_raw = self.dataset['image'][index]
        if img_raw.mode != 'RGB':
            img_raw = img_raw.convert('RGB')
            
        img_np = np.array(img_raw).astype(np.float32)
        img_tensor = torch.from_numpy(img_np)
        img = (img_tensor / 127.5) - 1.0 #normalization
        img = img.view(last_pn,ps,last_pn,ps,img.shape[-1])
        img = img.permute(0,2,1,3,4).contiguous()
        img = img.view(last_pn,last_pn,-1) # (num_patch,num_patch,768) for patch of size 16x6x3
        img = img.permute(2,0,1).unsqueeze(0)
        
        label_b = self.dataset['label'][index]
        x_BLCv_wo_first_l = []
        for pn in self.patch_nums:
            residual = F.interpolate(img,size=(pn,pn),mode='bicubic')
            x_BLCv_wo_first_l.append(residual.squeeze(0).reshape(pn**2,-1))
            reconstructed = F.interpolate(residual,size=(last_pn,last_pn),mode='bicubic')
            img = img - reconstructed

        x_BLCv_wo_first_l = torch.cat(x_BLCv_wo_first_l, dim=0)
        return x_BLCv_wo_first_l, label_b
