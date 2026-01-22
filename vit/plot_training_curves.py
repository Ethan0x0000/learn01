import argparse
import json
import os

import matplotlib.pyplot as plt


def load_jsonl(path):
    epochs = []
    losses = []
    accs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            epochs.append(data.get("epoch"))
            losses.append(data.get("loss"))
            accs.append(data.get("acc"))
    return epochs, losses, accs


def plot_curves(train_log, val_log, output_path, title=None):
    train_epochs, train_loss, train_acc = load_jsonl(train_log)
    val_epochs, val_loss, val_acc = load_jsonl(val_log) if val_log else ([], [], [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_epochs, train_loss, label="train")
    if val_loss:
        axes[0].plot(val_epochs, val_loss, label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(train_epochs, train_acc, label="train")
    if val_acc:
        axes[1].plot(val_epochs, val_acc, label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from jsonl logs")
    parser.add_argument("--train-log", type=str, required=True)
    parser.add_argument("--val-log", type=str, default=None)
    parser.add_argument("--output", type=str, default="training_curves.png")
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    output_path = plot_curves(args.train_log, args.val_log, args.output, args.title)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
