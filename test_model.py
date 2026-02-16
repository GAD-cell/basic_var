from src.models.var import VAR
import torch 
import numpy as np
import matplotlib.pyplot as plt 
from src.utils import TinyImageDataset
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
patch_nums = [1,2,3,4]
patch_num_tot = np.sum(np.array(patch_nums[1:])**2)
patch_size = 16
img_size = 64 # let's say imageNet-64
num_channels = 3
embed_dim = num_channels*(patch_size**2)
b_size = 3
num_classes = 200

model = VAR(num_classes=num_classes, 
            num_channels=num_channels, 
            patch_nums=patch_nums, 
            pixel_dim=embed_dim,
            man_dim=64, # 'manifold' dim  
            embed_dim=embed_dim)

label_b = torch.randint(200,(b_size,))
x_BLCv_wo_first_l = torch.zeros((b_size,patch_num_tot,embed_dim)) #(b,num_patch_tot,embed_dim)

img, latent_rep= model.forward(label_b,x_BLCv_wo_first_l)

print(img.shape,latent_rep.shape)


inference = model.autoregressive_infer_cfg(B=3) #(b,c,h,w)
print(inference.shape)
plt.imshow(inference[0].permute(1,2,0).cpu().numpy())
plt.axis('off')
plt.show()

dataset = TinyImageDataset()
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
scale_seq,label_b = next(iter(dataloader))
x_BLCv_wo_first_l = scale_seq[:,patch_nums[0]**2:]

scale_seq_pred, latent_rep= model.forward(label_b,x_BLCv_wo_first_l)
print(scale_seq_pred.shape,latent_rep.shape)

loss = torch.nn.MSELoss()(scale_seq_pred, scale_seq.to(device))
print(loss)
patches = scale_seq_pred[0, -16:, :]
patches_reshaped = patches.view(16, 3, 16, 16)
grid = patches_reshaped.view(4, 4, 3, 16, 16)
final_image = grid.permute(2, 0, 3, 1, 4).contiguous().view(3, 64, 64)
#plt.imshow(final_image.permute(1, 2, 0).detach().cpu().numpy())
#plt.axis('off')
#plt.show()

