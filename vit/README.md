# Vision Transformer (ViT) Project

This directory contains a custom implementation of the Vision Transformer (ViT) architecture, along with evaluation scripts and metric utilities.

## Project Structure

The project consists of the following key files:

### Core Implementation
- **`vit.py`**: Contains the `VisionTransformer` class, a custom implementation of the ViT architecture. It supports configurable image size, patch size, embedding dimension, depth, and number of heads.
- **`modules.py`**: Provides the building blocks for the Transformer:
  - `PatchEmbed`: Converts images into patch embeddings.
  - `MultiHeadSelfAttention`: Implements the self-attention mechanism.
  - `EncoderBlock`: Defines a single Transformer encoder layer with attention and MLP.

### Evaluation & Metrics
- **`metrics.py`**: A comprehensive library of metrics:
  - **Classification**: Accuracy, Confusion Matrix, Precision, Recall, F1 Score (macro/micro).
  - **Object Detection**: IoU (Intersection over Union), mAP (mean Average Precision).
- **`inference.py`**: An inference script that uses the `timm` library to load a pre-trained ViT model (`vit_base_patch16_224`) and evaluate it on the CIFAR-10 dataset.
- **`eval_classification_metrics.py`**: An advanced evaluation script that runs inference and computes detailed classification metrics using the utilities in `metrics.py`.

### Notebooks
- **`vit_train.ipynb`**: Complete training pipeline for the custom ViT model on CIFAR-10.
- **`vit_inference.ipynb`**: Interactive notebook for running inference with `timm` pre-trained models.
- **`vit_test.ipynb`**: A step-by-step educational notebook that implements and tests each ViT component from scratch.

### Configuration
- **`configs/vit_base.yaml`**: A YAML configuration file defining hyperparameters for training a ViT model on CIFAR-10 (e.g., `img_size: 32`, `patch_size: 4`, `depth: 6`).

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Custom ViT Model
You can use the custom `VisionTransformer` implementation in your code:

```python
from vit import VisionTransformer

# Initialize a standard ViT model
model = VisionTransformer(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    num_classes=1000
)
```

### Running Inference (timm)
To evaluate the pre-trained `timm` model on CIFAR-10:

```bash
python inference.py
```

### Calculating Metrics
To obtain detailed performance metrics (Precision, Recall, F1, Confusion Matrix):

```bash
python eval_classification_metrics.py
```
