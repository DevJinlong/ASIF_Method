# ASIF Method

ASIF is a camera-trap species recognition method that uses a dynamic species allowlist to determine which predictions can be accepted automatically. A prediction is accepted when its predicted species is on the allowlist and its confidence is at least 0.95.

## Code

| File | Purpose |
| --- | --- |
| [`code/main.py`](code/main.py) | Trains and validates the model and generates the species allowlist. |
| [`code/test.py`](code/test.py) | Runs inference and filters predictions using the allowlist and confidence threshold. |
| [`code/calculate_metrics.py`](code/calculate_metrics.py) | Computes evaluation metrics from prediction results. |

## Model

The trained model checkpoint is available at [`model/Step1/model.pth`](model/Step1/model.pth). The [`model/Step1/`](model/Step1/) directory also contains the model configuration, class index mapping, and evaluation outputs.

The checkpoint and large result files are stored with Git LFS. After cloning the repository, run the following command from the repository root to retrieve them.

```bash
git lfs pull
```

## CSV Files

| File | Purpose |
| --- | --- |
| [`csv/train_event.csv`](csv/train_event.csv) | Training split. |
| [`csv/val_event.csv`](csv/val_event.csv) | Validation split. |
| [`csv/test_event.csv`](csv/test_event.csv) | Test split. |
| [`model/Step1/results.csv`](model/Step1/results.csv) | Test predictions before filtering. |
| [`model/Step1/results_filtered.csv`](model/Step1/results_filtered.csv) | Predictions accepted by the allowlist and confidence threshold. |
| [`model/Step1/candidate_list.csv`](model/Step1/candidate_list.csv) | Candidate species considered when building the allowlist. |
| [`model/Step1/gate/gate_table.csv`](model/Step1/gate/gate_table.csv) | Species-level validation metrics used to apply the allowlist criteria. |
| [`model/Step1/gate/gate_selected.csv`](model/Step1/gate/gate_selected.csv) | Records for species selected by the validation criteria. |
| [`model/Step1/gate/allow_list.csv`](model/Step1/gate/allow_list.csv) | Final list of species eligible for automatic acceptance. |
| [`model/Step1/metrics.csv`](model/Step1/metrics.csv) | Evaluation metrics calculated from the prediction results. |
| [`model/integrated_f1_scores.csv`](model/integrated_f1_scores.csv) | Final list of species eligible for automatic acceptance. |
