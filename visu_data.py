from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

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
        
        # Normalisation : [0, 255] -> [-1, 1]
        img_np = np.array(img_raw).astype(np.float32)
        img_tensor = torch.from_numpy(img_np)
        img_normalized = (img_tensor / 127.5) - 1.0  # (64, 64, 3)
        
        # Convertir en format (C, H, W) pour interpolation DANS L'ESPACE PIXEL
        img_CHW = img_normalized.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 64, 64)
        
        label_b = self.dataset['label'][index]
        x_BLCv_wo_first_l = []
        
        img_residual = img_CHW.clone()
        
        # Pour chaque niveau de résolution
        for pn in self.patch_nums:
            # Downsampler dans l'espace PIXEL
            target_size = pn * ps  # Ex: pour pn=2, on veut 32×32 pixels
            residual_pixels = F.interpolate(img_residual, size=(target_size, target_size), 
                                           mode='bicubic', align_corners=False)
            
            # Patchifier le résidu à cette résolution
            # (1, 3, target_size, target_size) -> (pn, pn, 3*ps*ps)
            residual_patches = residual_pixels.squeeze(0)  # (3, target_size, target_size)
            residual_patches = residual_patches.reshape(3, pn, ps, pn, ps)  # (3, pn, ps, pn, ps)
            residual_patches = residual_patches.permute(1, 3, 0, 2, 4)  # (pn, pn, 3, ps, ps)
            residual_patches = residual_patches.reshape(pn, pn, 3 * ps * ps)  # (pn, pn, 768)
            
            # Stocker : (pn*pn, 768)
            x_BLCv_wo_first_l.append(residual_patches.reshape(pn * pn, -1))
            
            # Upsampler pour reconstruction DANS L'ESPACE PIXEL
            reconstructed_pixels = F.interpolate(residual_pixels, size=(last_pn * ps, last_pn * ps), 
                                                 mode='bicubic', align_corners=False)
            
            # Soustraire la reconstruction du résidu
            img_residual = img_residual - reconstructed_pixels
        
        # Concaténer tous les niveaux : (30, 768)
        x_BLCv_wo_first_l = torch.cat(x_BLCv_wo_first_l, dim=0)
        
        return x_BLCv_wo_first_l, label_b


def visualize_dataset_output(dataset, index=0, save_path='dataset_reconstruction.png'):
    """
    Visualise ce que le dataset produit réellement :
    - Récupère les 30 tokens du dataset
    - Les sépare par niveau [1, 2, 3, 4]
    - Dépatchifie et visualise chaque niveau
    - Montre la reconstruction progressive
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    
    last_pn = dataset.patch_nums[-1]
    ps = dataset.patch_size
    
    # Image originale pour comparaison
    img_raw = dataset.dataset['image'][index]
    if img_raw.mode != 'RGB':
        img_raw = img_raw.convert('RGB')
    
    img_np = np.array(img_raw).astype(np.float32)
    img_original = (img_np / 127.5) - 1.0
    
    # Récupérer ce que le dataset produit
    scale_seq, label = dataset[index]  # (30, 768)
    
    print(f"\n{'='*60}")
    print(f"Dataset Output Analysis")
    print(f"{'='*60}")
    print(f"scale_seq shape: {scale_seq.shape}")
    print(f"Label: {label}")
    print(f"Patch nums: {dataset.patch_nums}")
    print(f"Tokens per level: {[pn**2 for pn in dataset.patch_nums]}")
    print(f"{'='*60}\n")
    
    fig, axes = plt.subplots(2, len(dataset.patch_nums) + 1, 
                             figsize=(4 * (len(dataset.patch_nums) + 1), 8))
    
    # Original
    axes[0, 0].imshow((img_original + 1) / 2)
    axes[0, 0].set_title('Original Image\n(64×64)', fontsize=10)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Accumulateur pour la reconstruction progressive
    accumulated_pixels = torch.zeros(1, 3, last_pn * ps, last_pn * ps)
    
    # Séparer les tokens par niveau
    start_idx = 0
    for i, pn in enumerate(dataset.patch_nums):
        num_tokens = pn ** 2
        end_idx = start_idx + num_tokens
        
        # Extraire les tokens de ce niveau
        level_tokens = scale_seq[start_idx:end_idx]  # (pn*pn, 768)
        
        print(f"Level {i+1} ({pn}×{pn}): tokens shape = {level_tokens.shape}")
        
        # Reshape en patches: (pn*pn, 768) -> (pn, pn, 768)
        level_patches = level_tokens.reshape(pn, pn, 768)
        
        # Dépatchifier: (pn, pn, 768) -> (pn, pn, 3, 16, 16)
        level_patches = level_patches.reshape(pn, pn, 3, ps, ps)
        
        # Réarranger en image: (pn, pn, 3, 16, 16) -> (3, pn*16, pn*16)
        level_image = level_patches.permute(2, 0, 3, 1, 4).reshape(3, pn * ps, pn * ps)
        level_image = level_image.unsqueeze(0)  # (1, 3, pn*ps, pn*ps)
        
        # Interpoler à la taille finale si nécessaire
        if pn != last_pn:
            level_image_upscaled = F.interpolate(
                level_image, 
                size=(last_pn * ps, last_pn * ps), 
                mode='bicubic', 
                align_corners=False
            )
        else:
            level_image_upscaled = level_image
        
        # Accumuler
        accumulated_pixels += level_image_upscaled
        
        # Visualiser le résidu de ce niveau (avant upscale)
        res_vis = level_image.squeeze(0).permute(1, 2, 0).numpy()
        res_vis = np.clip((res_vis + 1) / 2, 0, 1)
        
        axes[0, i + 1].imshow(res_vis)
        axes[0, i + 1].set_title(
            f'Level {i+1}: {pn}×{pn}\n'
            f'Tokens: {num_tokens}\n'
            f'Size: {pn*ps}×{pn*ps}px', 
            fontsize=10
        )
        axes[0, i + 1].axis('off')
        
        # Visualiser la reconstruction accumulée
        acc_vis = accumulated_pixels.squeeze(0).permute(1, 2, 0).numpy()
        acc_vis = np.clip((acc_vis + 1) / 2, 0, 1)
        
        axes[1, i + 1].imshow(acc_vis)
        axes[1, i + 1].set_title(f'Accumulated\nup to level {i+1}', fontsize=10)
        axes[1, i + 1].axis('off')
        
        start_idx = end_idx
    
    # Calculer l'erreur de reconstruction finale
    final_reconstruction = accumulated_pixels.squeeze(0).permute(1, 2, 0).numpy()
    mse = np.mean((img_original - final_reconstruction) ** 2)
    psnr = 10 * np.log10(4.0 / mse) if mse > 0 else float('inf')
    
    print(f"\nReconstruction Quality:")
    print(f"  MSE: {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    
    if mse < 0.01:
        print(f"  ✓ Excellent reconstruction!")
    elif mse < 0.1:
        print(f"  ✓ Good reconstruction")
    else:
        print(f"  ✗ Poor reconstruction - there may be a problem")
    
    print(f"{'='*60}\n")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return mse, psnr


def compare_dataset_versions(old_dataset, new_dataset, index=0):
    """
    Compare deux versions du dataset côte à côte
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Ancien dataset
    try:
        old_tokens, old_label = old_dataset[index]
        axes[0, 0].set_title(f'Old Dataset\nShape: {old_tokens.shape}', fontsize=12)
        axes[0, 0].text(0.5, 0.5, f'Tokens: {old_tokens.shape[0]}\nLabel: {old_label}',
                       ha='center', va='center', fontsize=14)
        axes[0, 0].axis('off')
        
        # Visualiser l'ancien
        visualize_dataset_output(old_dataset, index, save_path='old_dataset_vis.png')
    except Exception as e:
        axes[0, 0].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
        axes[0, 0].axis('off')
    
    # Nouveau dataset
    try:
        new_tokens, new_label = new_dataset[index]
        axes[0, 1].set_title(f'New Dataset\nShape: {new_tokens.shape}', fontsize=12)
        axes[0, 1].text(0.5, 0.5, f'Tokens: {new_tokens.shape[0]}\nLabel: {new_label}',
                       ha='center', va='center', fontsize=14)
        axes[0, 1].axis('off')
        
        # Visualiser le nouveau
        visualize_dataset_output(new_dataset, index, save_path='new_dataset_vis.png')
    except Exception as e:
        axes[0, 1].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
        axes[0, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_comparison.png', dpi=150)
    plt.show()


# Utilisation
if __name__ == "__main__":
    from src.utils import TinyImageDataset
    
    # Créer le dataset
    dataset = TinyImageDataset(
        data_path='zh-plus/tiny-imagenet',
        patch_nums=[1, 2, 3, 4],
        patch_size=16
    )
    
    print("="*60)
    print("Testing Dataset Visualization")
    print("="*60)
    
    # Visualiser plusieurs exemples
    for idx in range(3):
        print(f"\n--- Sample {idx} ---")
        mse, psnr = visualize_dataset_output(
            dataset, 
            index=idx, 
            save_path=f'dataset_output_sample_{idx}.png'
        )