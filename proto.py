from datasets import load_dataset
from src.utils import TinyImageDataset
from torch.utils.data import DataLoader

def main():
    dataset = TinyImageDataset()
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)


if __name__ == "__main__":
    main()