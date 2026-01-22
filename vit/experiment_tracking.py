import argparse
import json
import os
from datetime import datetime

import yaml
from torch.utils.tensorboard import SummaryWriter


def create_experiment_dir(base_dir="./logs", exp_name=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = exp_name or f"exp_{timestamp}"
    exp_dir = os.path.join(base_dir, name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_config_json(config, output_dir, filename="config.json"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=True, indent=2)
    return path


def save_config_yaml(config, output_dir, filename="config.yaml"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)
    return path


def setup_tensorboard(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


def build_example_config():
    return {
        "model": {
            "name": "vit_base_patch16_224",
            "img_size": 224,
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "num_classes": 10,
        },
        "train": {
            "batch_size": 128,
            "epochs": 10,
            "lr": 1e-4,
            "weight_decay": 1e-4,
        },
        "data": {
            "dataset": "CIFAR-10",
            "data_root": "./data",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment record utilities")
    parser.add_argument("--base-dir", type=str, default="./logs")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--write-template", action="store_true")
    args = parser.parse_args()

    exp_dir = create_experiment_dir(args.base_dir, args.exp_name)
    config = build_example_config()
    json_path = save_config_json(config, exp_dir)
    yaml_path = save_config_yaml(config, exp_dir)

    writer = setup_tensorboard(os.path.join(exp_dir, "tensorboard"))
    writer.add_text("run_info", f"Config saved: {json_path}, {yaml_path}")
    writer.close()

    if args.write_template:
        template_path = os.path.join(exp_dir, "experiment_record.md")
        with open(template_path, "w", encoding="utf-8") as f:
            f.write("# Experiment Record\n\nFill in experiment details here.\n")
        print(f"Template written: {template_path}")

    print(f"Experiment directory: {exp_dir}")


if __name__ == "__main__":
    main()
