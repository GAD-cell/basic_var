from datasets import load_dataset
from src.utils import TinyImageDataset

def main():
    dataset = TinyImageDataset()

    dataset.__getitem__(0)


if __name__ == "__main__":
    main()