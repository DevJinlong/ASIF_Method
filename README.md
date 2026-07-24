# The Core–Edge Paradigm：

## A Habitat-Aware Framework for Deploying Deep Learning Species Recognition Across Heterogeneous Ecosystems

This repository contains the implementation and supporting files for the **Core–Edge adaptive framework**, a habitat-aware approach for deploying camera-trap species recognition models under temporal and geographic distribution shifts.

> **Double-blind review notice:** Author names, affiliations, contact details, and citation metadata are intentionally omitted during peer review.

## Overview

Camera traps are indispensable for wildlife monitoring, but the large volume of images they generate makes manual processing increasingly impractical. Deep learning models can achieve high accuracy on benchmark datasets, yet models trained in one ecosystem often lose reliability when transferred to another.

The Core–Edge paradigm separates the recognition system into two components:

- **Core:** a once-trained model based on ResNeXt-50, event-level attention, supervised contrastive learning, and temperature calibration. It learns robust visual representations from the source domain.
- **Edge:** a lightweight, deployment-specific dynamic allowlist. It identifies species for which the Core satisfies predefined reliability requirements and automatically accepts only predictions that meet both the allowlist and confidence conditions. All remaining images are flagged for manual review.

The Core remains stable, whereas the Edge can be updated using a small amount of labeled target-domain data. This design provides an explicit trade-off between automation breadth and prediction reliability.

## Framework

For each camera-trap capture event, valid images are processed by the ResNeXt-50 backbone. Event-level attention integrates contextual information among images from the same event. A linear classifier produces species logits, while a projection head supports supervised contrastive learning during training. Temperature scaling is fitted on the validation set to calibrate prediction confidence.

A species is included in the dynamic allowlist only when its validation performance meets the predefined reliability and coverage criteria. During inference, a prediction is automatically accepted only when:

1. the predicted species belongs to the current allowlist; and
2. the calibrated confidence is at least the configured confidence threshold.

Predictions that fail either condition are retained for manual identification.

## Datasets

### Snapshot Serengeti

Snapshot Serengeti (SS) was collected in Serengeti National Park, Tanzania, using 225 camera traps between July 2010 and August 2016. The complete dataset contains approximately 7.3 million images across 11 seasonal subsets (S1–S11), with an original image resolution of 2048 × 1536 pixels.

- **Source domain:** S1–S6 (July 2010–January 2013), containing approximately 1.2 million animal images from 47 species.
- **Temporal target domain:** S7–S11 (April 2013–May 2016), containing 59 species, including 12 species absent from S1–S6.

S1–S6 are partitioned by capture event into training, validation, and in-sample test sets at an 8:1:1 ratio. Event-level partitioning prevents images from the same event from appearing in different splits. The validation set is used for confidence calibration and allowlist construction.

### Snapshot Safari 2024 Expansion

The Snapshot Safari 2024 Expansion (Safari2024) dataset aggregates camera-trap sequences from 15 protected areas across Southern and Eastern Africa. It covers 1,824 camera stations and contains approximately 4.03 million images from 2.39 million capture events, representing 151 categories.

Safari2024 is used as an independent external test set for evaluating cross-regional generalization. It is not used to train the Core model or construct the source-domain allowlist. Stations are assigned to three habitat groups for habitat-stratified evaluation:

- arid/semi-arid savanna;
- transitional savanna; and
- humid savanna.

The datasets are not redistributed in this repository. Users should obtain them from their official sources and comply with the corresponding licenses and terms of use.

## Repository Structure

```text
ASIF_Method/
├── code/
│   ├── train.py                 # Model training and validation
│   ├── test.py                  # Calibrated inference and allowlist filtering
│   └── calculate_metrics.py     # Performance and automation metrics
├── csv/
│   ├── train_event.csv          # Event-level training split
│   ├── val_event.csv            # Event-level validation split
│   └── test_event.csv           # Event-level test split
├── pth/
│   └── Step1/
│       ├── model.pth            # Trained weights (Git LFS)
│       ├── label_to_index.json  # Species-to-index mapping
│       ├── model_T.json         # Learned calibration temperature
│       ├── model_config.json    # Saved model configuration
│       ├── metrics.csv          # Per-species metrics
│       └── gate/                # Allowlist and coverage outputs
├── README.md
└── README_CN.md
```

## Environment

The reported experiments used Python 3.x, PyTorch 2.5.1, CUDA, and NVIDIA RTX 4090 GPUs. Main dependencies include `torch`, `torchvision`, `numpy`, `pandas`, `Pillow`, `matplotlib`, `scikit-learn`, and `tqdm`.

```bash
pip install torch torchvision numpy pandas pillow matplotlib scikit-learn tqdm
```

## Data Preparation

Each CSV file must contain at least the following columns:

| Column | Description |
|---|---|
| `filename` | Path to the image file |
| `label` | Ground-truth species label |
| `event_id` | Capture-event identifier |

Before running the code, update the dataset and output paths in the scripts to match the local environment. Development-server paths are not portable.

## Training Configuration

All models were initialized with ImageNet-21k pre-trained weights. Training used AdamW with an initial learning rate of `1e-4`, weight decay of `1e-3`, and a batch size of `128`. The schedule consisted of a five-epoch linear warmup followed by StepLR decay with a factor of `0.1` every 10 epochs. Training ran for at most 60 epochs. Early stopping was triggered when validation accuracy did not improve for five consecutive epochs, and the best-performing checkpoint was retained.

## Usage

Run the scripts from the `code` directory in the following order:

```bash
cd code
python train.py
python test.py
python calculate_metrics.py
```

### Training

`train.py` trains the Core model, selects the best validation checkpoint, performs temperature calibration, and produces the information required for allowlist construction.

### Testing

`test.py` loads the trained model and calibration parameters, performs event-aware inference, and applies the allowlist and calibrated-confidence acceptance rules.

### Metric Calculation

`calculate_metrics.py` calculates per-species Recall, Precision, F1-score, allowlist statistics, and the overall automation rate.

## Evaluation Scenarios

1. **In-sample evaluation:** event-level test split from Snapshot Serengeti S1–S6.
2. **Temporal generalization:** transfer from S1–S6 to S7–S11 within the Serengeti ecosystem.
3. **Geographic generalization:** transfer from Snapshot Serengeti to Safari2024 protected areas across heterogeneous habitats.

For temporal adaptation, 10% of the new target-domain data is labeled and used to update the Edge. The remaining 90% is retained for evaluation.

## Main Results

| Evaluation setting | Automation rate | Main observation |
|---|---:|---|
| Snapshot Serengeti S1–S6 | 78.59% | 18 allowlisted species exceeded 95% F1-score |
| Snapshot Safari 2024 Expansion | 29.07% | Geographic transfer caused a substantial decrease |
| Arid/semi-arid savanna | 10.09% | Lowest transferability |
| Transitional savanna | 24.51% | Intermediate transferability |
| Humid savanna | 45.13% | Highest transferability |

Under temporal shifts from S1–S6 to S7–S11, updating the Edge with 10% of the new data restored the required reliability without retraining the Core. Geographic evaluation revealed a habitat gradient, indicating that source–target habitat similarity is an important determinant of model transferability.

## Model Weights

The trained checkpoint is stored at `pth/Step1/model.pth` and tracked with Git Large File Storage (Git LFS).

```bash
git lfs install
git clone <repository-url>
```

## Reproducibility Notes

- Perform all data partitions at the capture-event level.
- Never place the same event in more than one split.
- Use validation data for temperature calibration and allowlist construction.
- Do not use test data to train the Core or construct the evaluated allowlist.
- Use calibrated probabilities when applying the confidence threshold.
- Check random seeds and local data paths before reproducing an experiment.

## Citation

Citation information will be added after peer review and publication.

## License

License information will be added after peer review. Dataset usage remains subject to the licenses and terms of the original dataset providers.
