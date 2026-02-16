from src.models.var import VAR
import torch 
import numpy as np
import matplotlib.pyplot as plt 

patch_nums = [1,2,3,4]
patch_num_tot = np.sum(np.array(patch_nums[1:])**2)
patch_size = 16
img_size = 64 # let's say imageNet-64
num_channels = 3
embed_dim = num_channels*(patch_size**2)
b_size = 3
num_classes = 200

model = VAR(num_classes=200, 
            num_channels=num_channels, 
            patch_nums=patch_nums, 
            pixel_dim=embed_dim, 
            embed_dim=embed_dim)

label_b = torch.randint(200,(b_size,))
x_BLCv_wo_first_l = torch.zeros((b_size,patch_num_tot,embed_dim)) #(b,num_patch_tot,embed_dim)

latent_rep, img = model.forward(label_b,x_BLCv_wo_first_l)

print(img.shape,latent_rep.shape)


inference = model.autoregressive_infer_cfg(B=3) #(b,c,h,w)
print(inference.shape)

#plt.imshow(inference[0,:,:].permute(1,2,0).cpu().numpy())
#plt.show()