import argparse

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_train_transforms(img_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def build_eval_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def create_cifar10_dataloader(
    data_root="./data",
    batch_size=128,
    img_size=224,
    num_workers=2,
    train=True,
):
    if train:
        transform = build_train_transforms(img_size)
    else:
        transform = build_eval_transforms(img_size)
    dataset = datasets.CIFAR10(
        root=data_root,
        train=train,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
    )


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 dataloader with augmentation")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    loader = create_cifar10_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        train=True,
    )
    images, labels = next(iter(loader))
    print(f"Loaded batch: images={images.shape}, labels={labels.shape}")


if __name__ == "__main__":
    main()
