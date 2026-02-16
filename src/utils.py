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
        
        img = torch.from_numpy(np.array(self.dataset['image'][index])) #(h,w,c)
        img = img.view(last_pn,ps,last_pn,ps,img.shape[-1])
        img = img.permute(0,2,1,3,4).contiguous()
        img = img.view(last_pn,last_pn,-1) # (num_patch,num_patch,768) for patch of size 16x6x3
        img = img.permute(2,0,1).unsqueeze(0)
        x_BLCv_wo_first_l = []
        label_b = self.dataset['label'][index]

        for pn in self.patch_nums:
            residual = F.interpolate(img,size=(pn,pn),mode='bicubic')
            x_BLCv_wo_first_l.append(residual.squeeze(0).reshape(pn**2,-1))
            reconstructed = F.interpolate(residual,size=(last_pn,last_pn),mode='bicubic')
            img = img - reconstructed
        
        return x_BLCv_wo_first_l, label_b
