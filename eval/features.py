import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_dinov2_model(device: torch.device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    return model


def _dinov2_tfm():
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def extract_dinov2_features(model, imgs_bchw: torch.Tensor) -> torch.Tensor:
    tfm = _dinov2_tfm()
    imgs = tfm(imgs_bchw)
    with torch.no_grad():
        feats = model(imgs).detach()
    return feats


def extract_dinov2_features_with_grad(
    model,
    imgs_bchw: torch.Tensor,
    freeze_model: bool = True,
) -> torch.Tensor:
    """
    Grad-enabled DINOv2 features for generated images.
    Use freeze_model=True to avoid updating DINO weights while keeping gradients
    flowing to imgs_bchw.
    """
    if freeze_model:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    tfm = _dinov2_tfm()
    imgs = tfm(imgs_bchw)
    feats = model(imgs)
    return feats

# Create new .pt file for storing image features
def preprocess_dataset(dataset_path : Path, device: torch.device, batch_size: int = 64):
    dataset = datasets.ImageFolder(dataset_path, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = get_dinov2_model(device)
    all_features = []
    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        features = extract_dinov2_features(model, imgs)
        all_features.append(features.cpu())
    all_features = torch.cat(all_features, dim=0)
    torch.save(all_features, dataset_path.parent / f"{dataset_path.stem}_features.pt")

if __name__ == "__main__":
    device = pick_device()
    preprocess_dataset(Path("data/ImageNet"), device)
