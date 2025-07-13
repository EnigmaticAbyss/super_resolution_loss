# Super Resolution Loss

A PyTorch-based framework for training and evaluating image super-resolution models with various custom loss functions.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [SLURM Job](#slurm-job)
- [Project Structure](#project-structure)
- [Loss Functions](#loss-functions)
- [Supported Models](#supported-models)
- [Metrics](#metrics)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Features

- Implement multiple perceptual and frequency-based loss functions
- Support for ESRGAN, EDSR, NinaSR, and SwinIR models
- Easy configuration via YAML files
- Training with TensorBoard logging and early stopping
- Evaluation with PSNR, SSIM, FID, and LPIPS metrics
- Visualization utilities for comparing results

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/EnigmaticAbyss/super_resolution_loss.git
   cd super_resolution_loss
   ```
2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate super_resolution_loss
   ```

*Note: Ensure you have CUDA and PyTorch installed for GPU acceleration.*

## Usage

### Configuration

Create your experiment configuration file(s) under `config/`, for example `config/config_experiment1.yml`:

```yaml
model: ESRGANGenerator
loss_function: PerceptualLoss
train_hr_path: path/to/train/HR
train_lr_path: path/to/train/LR
valid_hr_path: path/to/valid/HR
valid_lr_path: path/to/valid/LR
batch_size: 16
learning_rate: 1e-4
epochs: 100
log_dir: logs/tensorboard
model_save_dir: saved_models
early_stopping_patience: 10
```

### Training

Run all experiments defined in `config/`:

```bash
python main.py
```

Or train a single experiment:

```bash
python scripts/train.py config/config_experiment1.yml
```

### Evaluation

Evaluate a trained model using its config file:

```bash
python scripts/eval.py config/config_experiment1.yml
```

Or directly from a saved `.pth` checkpoint:

```bash
python scripts/eval.py saved_models/ESRGANGenerator_PerceptualLoss.pth
```

### SLURM Job

A sample SLURM submission script `run.sh` is provided:

```bash
sbatch run.sh
```

## Project Structure

```
├── config/                # YAML configuration files for experiments
├── losses/                # Custom loss function implementations
├── models/                # Model architectures (ESRGAN, etc.)
├── scripts/
│   ├── train.py           # Training loop
│   └── eval.py            # Evaluation and metrics
├── utils/
│   ├── dataset.py         # Data loading utilities
│   ├── metrics.py         # PSNR, SSIM, FID, LPIPS
│   └── visualization.py   # Result plotting functions
├── main.py                # Entry point for running experiments
├── environment.yml        # Conda environment specifications
├── run.sh                 # SLURM batch script
└── test.py                # (Optional) Test scripts
```

## Loss Functions

- `PerceptualLoss`
- `LPIPSLoss`
- `GradientLoss` (L2)
- `FrequencyLoss`
- `FourierPerceptualLoss`
- `FourierDifferencePerceptualLoss`
- `FourierFeaturePerceptualLoss`
- `CombinedLoss`
- `HieraPerceptualLoss`
- `HieraNoFreqPercep`
- `HieraNoFreqPercepNoMSE`
- `HieraPerceptualLossNoMSE`

## Supported Models

- ESRGANGenerator
- EDSR (from `torchsr`)
- NinaSR (`torchsr`)
- SwinIR

## Metrics

- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Fréchet Inception Distance (FID)
- Learned Perceptual Image Patch Similarity (LPIPS)

## Visualization

Use the `display_images_from_folders` utility to compare LR, HR, and SR outputs:

```python
from utils.visualization import display_images_from_folders
display_images_from_folders("test_results")
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

