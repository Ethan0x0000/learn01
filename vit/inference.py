import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm


def create_model(model_name="vit_base_patch16_224", num_classes=10, pretrained=True, device=None):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, device


def create_dataloader(data_root="./data", batch_size=128, img_size=224, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    pretrained = not args.no_pretrained
    model, device = create_model(
        model_name=args.model_name,
        num_classes=10,
        pretrained=pretrained,
        device=None,
    )
    dataloader = create_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )
    accuracy = evaluate(model, dataloader, device)
    print(f"Test accuracy on CIFAR-10: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

