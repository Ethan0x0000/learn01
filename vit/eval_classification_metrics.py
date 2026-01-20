import os
import sys

import torch

import metrics


def _get_inference_module():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    import inference

    return inference


def run_evaluation():
    inference = _get_inference_module()
    model, device = inference.create_model(
        model_name="vit_base_patch16_224",
        num_classes=10,
        pretrained=True,
        device=None,
    )
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    dataloader = inference.create_dataloader(
        data_root=data_root,
        batch_size=128,
        img_size=224,
        num_workers=2,
    )
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            all_logits.append(outputs.cpu())
            all_targets.append(targets.cpu())
    if len(all_logits) == 0:
        return None
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics_dict = metrics.classification_metrics_from_logits(
        logits, targets, num_classes=10, average="macro"
    )
    return metrics_dict


def main():
    metrics_dict = run_evaluation()
    if metrics_dict is None:
        print("No data to evaluate.")
        return
    print("Accuracy:", metrics_dict["accuracy"])
    print("Precision (macro):", metrics_dict["precision"])
    print("Recall (macro):", metrics_dict["recall"])
    print("F1 (macro):", metrics_dict["f1"])
    print("Confusion matrix shape:", tuple(metrics_dict["confusion_matrix"].shape))


if __name__ == "__main__":
    main()

