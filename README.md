# InfoBatch-PyTorch

A simplified PyTorch implementation of [InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning, ICML 2024](https://arxiv.org/abs/2303.04947).

## Overview

InfoBatch is a data pruning technique that speeds up training without accuracy loss by dynamically selecting informative samples during training. This implementation uses ResNet-18 on CIFAR-10/100 datasets.

**Key Features:**
- Dynamic data pruning based on per-sample loss scores
- Unbiased gradient estimation using importance sampling
- Multiple pruning strategies (standard, moving average, reverse)
- MLflow integration for experiment tracking

## Requirements

- Docker with GPU support
- NVIDIA GPU with CUDA 11.7
- Base image: `pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel`

## Quick Start

### 1. Setup Container

```sh
sh run_container.sh
```

**Note:** If the volume mount path is incorrect, modify the `-v` parameter in `run_container.sh`:
```sh
docker run -it -d \
  -v ~/info_batch:/sources \
  -v ~/info_batch/dataset/:/sources/dataset \
  --ipc=host --gpus=all \
  --name info_batch \
  pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
```

### 2. Install Dependencies

After running the container:
```sh
docker exec -it info_batch bash
pip install mlflow
```

### 3. Run Experiments

Inside the container:
```sh
cd /sources/src
python res18_cifar10_ib.py
```

## Project Structure

```
InfoBatch-Pytorch/
├── src/                          # Source code
│   ├── res18_cifar10_whole.py    # Baseline training (full dataset)
│   ├── res18_cifar10_ib.py       # InfoBatch with standard policy
│   ├── res18_cifar10_ib_ma.py    # InfoBatch with moving average threshold
│   ├── res18_cifar10_ib_rev.py   # InfoBatch with reverse policy (train pruned samples)
│   ├── res18_cifar100_*.py       # CIFAR-100 variants
│   └── utils/
│       ├── dataset.py            # Custom dataset classes (InfoCIFAR10/100)
│       ├── policy.py             # Standard pruning policy
│       ├── ma_policy.py          # Moving average policy
│       └── reverse_policy.py     # Reverse pruning policy
├── config/                       # Training configurations
│   ├── res18_cifar10_whole.py    # Hyperparameters for CIFAR-10
│   └── res18_cifar100_whole.py   # Hyperparameters for CIFAR-100
└── run_container.sh              # Docker container launch script
```

## Experiment Variants

| File | Description | Use Case |
|------|-------------|----------|
| `res18_cifarXX_whole.py` | Baseline training on full dataset | Comparison baseline |
| `res18_cifarXX_ib.py` | Standard InfoBatch with threshold-based pruning | Main method |
| `res18_cifarXX_ib_ma.py` | InfoBatch with moving average threshold | Smoother threshold updates |
| `res18_cifarXX_ib_rev.py` | Trains on pruned samples instead of kept samples | Ablation study |

## Configuration

### Hyperparameters

Default settings in `config/res18_cifar10_whole.py`:
```python
num_epochs = 100
batch_size = 128
max_lr = 0.1
momentum = 0.9
pct_start = 0.3         # OneCycleLR parameter
weight_decay = 5e-4
label_smooth = 0.0
```

### Pruning Policy

Adjustable in `src/utils/policy.py`:
```python
PruningPolicy(
    data_size,           # Total dataset size
    total_epoch,         # Total training epochs
    prob=0.5,           # Pruning probability for low-score samples
    anneal=0.875        # Fraction of epochs to apply pruning (87.5%)
)
```

**How it works:**
1. Computes per-sample loss scores each iteration
2. At epoch end, calculates threshold as mean score of used samples
3. Divides data into:
   - **D2**: High-score samples (≥ threshold) → always used
   - **D1**: Low-score samples (< threshold) → probabilistically pruned
4. Applies importance sampling weights: `1/(1-prob)` for kept D1 samples
5. After annealing period, uses full dataset

## Experiment Tracking

All experiments log metrics to MLflow:
- Training/test loss and accuracy
- Learning rate schedule
- Epoch time and cumulative duration
- Dynamic threshold values
- Iteration count per epoch

### View Results

```sh
mlflow ui
```

Then open `http://localhost:5000` in your browser.

**Logged Metrics:**
- `train_loss`, `test_loss`, `test_accuracy`
- `epoch_time`, `train_duration`
- `threshold` (dynamic pruning threshold)
- `iter_count` (number of batches per epoch)

## Implementation Details

### Custom Dataset

`InfoCIFAR10/100` extends torchvision's CIFAR datasets to return:
- `img`: Image tensor
- `target`: Class label
- `index`: Sample index (for score tracking)
- `scaler`: Importance sampling weight

### Custom Sampler

`PruningPolicy` implements PyTorch's `Sampler` interface:
- Tracks per-sample scores
- Dynamically updates pruning decisions each epoch
- Returns `(index, scaler)` tuples for dataloader

### Training Loop

```python
for idx, (x, y, sample_idx, scaler) in train_loader:
    logits = model(x)
    loss = criterion(logits, y)  # Per-sample loss

    policy.update_scores(sample_idx, loss.detach())  # Track scores

    loss = (loss * scaler).mean()  # Apply importance weights
    loss.backward()
    optimizer.step()

policy.update_policy(epoch+1)  # Update pruning at epoch end
```

## Troubleshooting

### Docker Issues
- **GPU not detected:** Ensure `nvidia-docker` is installed and `--gpus=all` flag is set
- **Permission denied:** Run `chmod +x run_container.sh`

### Training Issues
- **Out of memory:** Reduce `batch_size` in config files
- **Dataset not found:** Datasets auto-download to `/sources/dataset/` on first run
- **MLflow not found:** Install inside container: `pip install mlflow`

### Path Issues
- All experiments expect `/sources` as the working directory
- Modify `sys.path.append('/sources')` if using different mount paths

## Citation

If you use this code, please cite:
```bibtex
@inproceedings{qin2024infobatch,
  title={InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```

## License

This is a research implementation. Please refer to the original paper for academic use.
