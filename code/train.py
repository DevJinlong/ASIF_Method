import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torch.optim.lr_scheduler import LambdaLR, StepLR
from tqdm import tqdm


# ============================================================
# Global Configuration
# ============================================================
batch_size = 40
num_worker = 6
GPU = "cuda:" + "0"
device = torch.device(GPU if torch.cuda.is_available() else "cpu")
print("device", device)

Recall_Precision = 95
confidence_threshold = 0.95
coverage_threshold = 0.70
ENABLE_PLOT = True

# ============================================================
# Contrastive Learning Configuration: Dynamic Linear Decay
# ============================================================
USE_CONTRASTIVE = True
LAMBDA_MCL = 0.4
LAMBDA_MCL_MIN = 0.0
TEMP_HARD = 0.1
MCL_FEAT_DIM = 128
PROJ_DROPOUT = 0.3

# ============================================================
# Event Attention Configuration
# ============================================================
ATTN_D_MODEL = 2048
ATTN_NHEAD = 8
ATTN_NUM_LAYERS = 2
ATTN_DIM_FEEDFORWARD = 4096
ATTN_DROPOUT = 0.1

MAX_IMAGES_PER_EVENT = None
ONLY_EVENTS_WITH_MULTI_IMAGES = False


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


set_seed(66)


# =========================
# transforms
# =========================
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}


# =============================
# Dataset: Grouped by event_id
# =============================
class CreateEventDatasetFromImages(Dataset):
    def __init__(
        self,
        csv_path,
        file_path,
        transform=None,
        label_map_path=None,
        save_map=False,
        training=False,
        max_images_per_event=None,
        only_events_with_multi_images=False,
    ):
        self.file_path = file_path
        self.data_info = pd.read_csv(csv_path)

        required_cols = ["filename", "label", "event_id"]
        for c in required_cols:
            if c not in self.data_info.columns:
                raise RuntimeError(f"{csv_path} Missing required column: {c}. End-to-end attention requires filename, label, and event_id.")

        self.data_info["filename"] = self.data_info["filename"].astype(str)
        self.data_info["label"] = self.data_info["label"].astype(str)
        self.data_info["event_id"] = self.data_info["event_id"].astype(str)

        self.transform = transform
        self.training = training
        self.max_images_per_event = max_images_per_event

        if label_map_path and os.path.exists(label_map_path):
            with open(label_map_path, "r") as f:
                self.label_to_index = json.load(f)
        else:
            unique_labels = np.unique(self.data_info["label"].astype(str).values)
            self.label_to_index = {str(label): int(idx) for idx, label in enumerate(unique_labels)}
            if save_map and label_map_path:
                os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
                with open(label_map_path, "w") as f:
                    json.dump(self.label_to_index, f, ensure_ascii=False, indent=2)

        missing_labels = sorted(set(self.data_info["label"].astype(str)) - set(self.label_to_index.keys()))
        if len(missing_labels) > 0:
            raise RuntimeError("CSV 中存在 label_to_index.json 没有的类别：\n" + "\n".join(missing_labels[:50]))

        self.groups = []
        for event_id, g in self.data_info.groupby("event_id", sort=False):
            g = g.reset_index(drop=True)
            if only_events_with_multi_images and len(g) < 2:
                continue
            self.groups.append((event_id, g))

        if len(self.groups) == 0:
            raise RuntimeError(f"No valid event groups found in {csv_path}")

        lens = [len(g) for _, g in self.groups]
        print(
            f"[EventDataset] {os.path.basename(csv_path)} | rows={len(self.data_info)} "
            f"events={len(self.groups)} min_len={min(lens)} max_len={max(lens)} mean_len={np.mean(lens):.2f}"
        )

    def __getitem__(self, index):
        event_id, g = self.groups[index]

        if self.max_images_per_event is not None and len(g) > self.max_images_per_event:
            if self.training:
                g = g.sample(n=self.max_images_per_event, replace=False).reset_index(drop=True)
            else:
                g = g.iloc[: self.max_images_per_event].reset_index(drop=True)

        imgs, labels, filenames = [], [], []

        for _, row in g.iterrows():
            single_image_name = str(row["filename"])
            single_image_path = os.path.join(self.file_path, single_image_name)
            img_as_img = Image.open(single_image_path).convert("RGB")

            if self.transform:
                img_as_img = self.transform(img_as_img)

            label = str(row["label"])
            label_index = int(self.label_to_index[label])

            imgs.append(img_as_img)
            labels.append(label_index)
            filenames.append(single_image_name)

        imgs = torch.stack(imgs, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        return imgs, labels, filenames, event_id

    def __len__(self):
        return len(self.groups)


# =========================
# Event
# =========================
def event_collate_fn(batch):
    batch_size_now = len(batch)
    max_len = max(item[0].shape[0] for item in batch)
    C, H, W = batch[0][0].shape[1:]

    images = torch.zeros(batch_size_now, max_len, C, H, W, dtype=batch[0][0].dtype)
    labels = torch.full((batch_size_now, max_len), -100, dtype=torch.long)
    mask = torch.zeros(batch_size_now, max_len, dtype=torch.bool)

    all_filenames = []
    all_event_ids = []

    for b, (imgs, labs, filenames, event_id) in enumerate(batch):
        n = imgs.shape[0]
        images[b, :n] = imgs
        labels[b, :n] = labs
        mask[b, :n] = True
        all_filenames.append(filenames)
        all_event_ids.append(event_id)

    return images, labels, mask, all_filenames, all_event_ids


# =========================
# ResNeXt50 + Event Attention + Projection Head
# =========================
class ResNeXt50EventAttentionContrastive(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feat_dim: int = 128,
        dropout: float = 0.3,
        attn_d_model: int = 2048,
        attn_nhead: int = 8,
        attn_num_layers: int = 2,
        attn_dim_feedforward: int = 4096,
        attn_dropout: float = 0.1,
    ):
        super().__init__()

        backbone = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.in_features = in_features

        if attn_d_model != in_features:
            raise ValueError(f"attn_d_model={attn_d_model} must equal backbone feature dim={in_features}")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_d_model,
            nhead=attn_nhead,
            dim_feedforward=attn_dim_feedforward,
            dropout=attn_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.event_attention = nn.TransformerEncoder(
            encoder_layer,
            num_layers=attn_num_layers,
        )

        self.classifier = nn.Linear(in_features, num_classes)

        # The contrastive learning projection head receives the context features produced by the attention module
        self.proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, feat_dim),
        )

        self._init_new_layers()

    def _init_new_layers(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, return_features: bool = False):

        B, N, C, H, W = x.shape

        # Feed only real images into the backbone to prevent padded images from affecting BatchNorm
        x_valid = x[mask]  # [M, C, H, W]
        feat_valid = self.backbone(x_valid)  # [M, 2048]

        feat = x.new_zeros(B, N, self.in_features)
        feat[mask] = feat_valid

        padding_mask = ~mask.bool()

        context_feat = self.event_attention(
            feat,
            src_key_padding_mask=padding_mask,
        )  # [B, N, 2048]

        logits = self.classifier(context_feat)  # [B, N, num_classes]

        if return_features:
            context_valid = context_feat[mask]  # [M, 2048]
            z = self.proj(context_valid)
            z = F.normalize(z, dim=1)
            return logits, z

        return logits


# =========================
# Supervised Contrastive Loss
# =========================
class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, eps: float = 1e-12):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)

        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        logits_mask = torch.ones_like(logits, device=device)
        logits_mask.fill_diagonal_(0)

        positive_mask = torch.eq(labels, labels.T).float().to(device)
        positive_mask = positive_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)

        pos_count = positive_mask.sum(dim=1)
        valid = pos_count > 0

        if valid.sum() == 0:
            return features.sum() * 0.0

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (pos_count + self.eps)
        loss = -mean_log_prob_pos[valid].mean()

        return loss


# =========================
# Dynamic Linear Decay Function
# =========================
def get_dynamic_lambda_mcl(epoch: int, total_epochs: int, lambda_start: float, lambda_min: float = 0.0):
    if total_epochs <= 1:
        return float(lambda_min)

    progress = (epoch - 1) / (total_epochs - 1)
    current = lambda_start * (1.0 - progress)

    return max(float(lambda_min), float(current))


# =========================
# Temperature scaling
# =========================
def calibrate_temperature(model, val_loader, device):
    model.eval()
    nll_criterion = nn.CrossEntropyLoss().to(device)

    logits_list, labels_list = [], []
    with torch.no_grad():
        for inputs, labels, mask, _, _ in tqdm(val_loader, desc="Collect logits for T"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            logits = model(inputs, mask)
            logits_list.append(logits[mask])
            labels_list.append(labels[mask])

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    log_T = nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_T], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T)
        loss = nll_criterion(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    temperature = float(torch.exp(log_T).detach().item())
    print(f"Calibrated temperature (init=1.0): {temperature:.4f}")
    return temperature


# =========================
# validate acc
# =========================
def validate_model(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels, mask, _, _ in tqdm(val_loader, desc="Validating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            outputs = model(inputs, mask)
            predicted = outputs.argmax(dim=-1)

            total += int(mask.sum().item())
            correct += int((predicted[mask] == labels[mask]).sum().item())

    return 100.0 * correct / max(1, total)


# =========================
# train + calibrate T
# =========================
def train_model(train_loader, val_loader, model_path, num_classes, patience=7):
    model = ResNeXt50EventAttentionContrastive(
        num_classes=num_classes,
        feat_dim=MCL_FEAT_DIM,
        dropout=PROJ_DROPOUT,
        attn_d_model=ATTN_D_MODEL,
        attn_nhead=ATTN_NHEAD,
        attn_num_layers=ATTN_NUM_LAYERS,
        attn_dim_feedforward=ATTN_DIM_FEEDFORWARD,
        attn_dropout=ATTN_DROPOUT,
    )
    model.to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_mcl = SupConLoss(temperature=TEMP_HARD)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    warmup_epochs = 5
    total_epochs = 60

    warmup_factor = lambda epoch: epoch / warmup_epochs if epoch <= warmup_epochs else 1

    scheduler_warmup = LambdaLR(optimizer, lr_lambda=warmup_factor)
    scheduler_step = StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0
    no_improve = 0

    for epoch in range(total_epochs):
        epoch_id = epoch + 1

        current_lambda_mcl = get_dynamic_lambda_mcl(
            epoch=epoch_id,
            total_epochs=total_epochs,
            lambda_start=LAMBDA_MCL,
            lambda_min=LAMBDA_MCL_MIN,
        )

        print(
            f"Epoch {epoch_id}/{total_epochs} | "
            f"lambda_mcl={current_lambda_mcl:.6f}"
        )

        model.train()

        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch_id}")

        for inputs, labels, mask, _, _ in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            if USE_CONTRASTIVE:
                logits, features = model(inputs, mask, return_features=True)

                valid_logits = logits[mask]
                valid_labels = labels[mask]

                loss_cls = criterion_cls(valid_logits, valid_labels)
                loss_mcl = criterion_mcl(features, valid_labels)

                loss = loss_cls + current_lambda_mcl * loss_mcl
            else:
                logits = model(inputs, mask)

                valid_logits = logits[mask]
                valid_labels = labels[mask]

                loss_cls = criterion_cls(valid_logits, valid_labels)
                loss_mcl = torch.tensor(0.0, device=device)

                loss = loss_cls

            loss.backward()
            optimizer.step()

        val_acc = validate_model(model, val_loader, device)

        print(f"Epoch {epoch_id}: Validation Accuracy = {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved at {model_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

        if epoch <= warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler_step.step()

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    temperature = calibrate_temperature(model, val_loader, device)

    temp_path = model_path.replace(".pth", "_T.json")
    with open(temp_path, "w") as f:
        json.dump({"temperature": temperature}, f, indent=2)

    return model, temperature


# =======================================
# Inference: Return results and results_t
# =======================================
def predict_val_data(model, val_loader, label_map_path, temperature):
    model.eval()
    model.to(device)

    with open(label_map_path, "r") as f:
        index_to_label = {int(v): k for k, v in json.load(f).items()}

    rows, rows_t = [], []

    with torch.no_grad():
        for inputs, labels, mask, all_filenames, all_event_ids in tqdm(val_loader, desc="Predicting (val)"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            logits = model(inputs, mask)
            probs = softmax(logits, dim=-1)
            probs_t = softmax(logits / float(temperature), dim=-1)

            pred = probs.argmax(dim=-1)
            pred_t = probs_t.argmax(dim=-1)

            B, N, num_classes = probs.shape

            for b in range(B):
                for n in range(N):
                    if not bool(mask[b, n].item()):
                        continue

                    actual = index_to_label[int(labels[b, n].item())]
                    filename = all_filenames[b][n]
                    event_id = all_event_ids[b]

                    row = {
                        "filename": filename,
                        "event_id": event_id,
                        "Actual": actual,
                        "Predicted": index_to_label[int(pred[b, n].item())],
                    }
                    row_t = {
                        "filename": filename,
                        "event_id": event_id,
                        "Actual": actual,
                        "Predicted": index_to_label[int(pred_t[b, n].item())],
                    }

                    for c in range(num_classes):
                        row[f"Prob_{c}"] = float(probs[b, n, c].item())
                        row_t[f"Prob_{c}"] = float(probs_t[b, n, c].item())

                    rows.append(row)
                    rows_t.append(row_t)

    results = pd.DataFrame(rows)
    results_t = pd.DataFrame(rows_t)

    prob_cols = sorted(
        [c for c in results.columns if c.startswith("Prob_")],
        key=lambda x: int(x.split("_")[1]),
    )

    results = results[["filename", "event_id", "Actual", "Predicted"] + prob_cols]
    results_t = results_t[["filename", "event_id", "Actual", "Predicted"] + prob_cols]

    return results, results_t


# ============================
# Filter Samples by Confidence
# ============================
def filter_confidence(data_path: str, output_path: str, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    prob_cols = [c for c in df.columns if c.startswith("Prob_")]
    if not prob_cols:
        raise ValueError("filter_confidence: no Prob_* columns found in input csv.")

    df["_maxprob"] = df[prob_cols].max(axis=1)
    df_filt = df[df["_maxprob"] >= float(threshold)].copy()
    df_filt.drop(columns=["_maxprob"], inplace=True)

    df_filt.to_csv(output_path, index=False)
    return df_filt


# =========================
# Calculate Metrics
# =========================
def calculate_metrics(data_path: str, metrics_path: str):
    df = pd.read_csv(data_path)
    if "Actual" not in df.columns or "Predicted" not in df.columns:
        raise ValueError(f"{data_path} must contain columns: Actual, Predicted")

    species = df["Actual"].unique()
    results = []

    for s in species:
        tp = int(((df["Actual"] == s) & (df["Predicted"] == s)).sum())
        fp = int(((df["Actual"] != s) & (df["Predicted"] == s)).sum())
        fn = int(((df["Actual"] == s) & (df["Predicted"] != s)).sum())
        tn = int(((df["Actual"] != s) & (df["Predicted"] != s)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        results.append(
            {
                "Species": s,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Recall": round(recall, 4) * 100,
                "Precision": round(precision, 4) * 100,
                "F1-Score": round(f1, 4) * 100,
            }
        )

    out = pd.DataFrame(results).sort_values(by="Species")
    if out.empty:
        out = pd.DataFrame(columns=["Species", "TP", "FP", "FN", "TN", "Recall", "Precision", "F1-Score"])
    else:
        out = out[["Species", "TP", "FP", "FN", "TN", "Recall", "Precision", "F1-Score"]]
    out.to_csv(metrics_path, index=False)


# =========================
# Create Folder
# =========================
def create_step_folder(step: int):
    step_path = f"Step{step}"
    os.makedirs(step_path, exist_ok=True)
    return step_path


# =========================
# Plot the Coverage Rates of Candidate Species
# =========================
def plot_gate_coverage_threshold(gate_table: pd.DataFrame, threshold: float, out_png: str):
    if gate_table is None or gate_table.empty:
        raise ValueError("plot_gate_coverage_threshold: gate_table empty")

    y = gate_table["Coverage"].to_numpy(dtype=float)
    x = np.arange(len(y))
    species = gate_table["Species"].astype(str).tolist()

    plt.figure(figsize=(max(10, len(y) * 0.6), 6))
    plt.plot(x, y, marker="o", linewidth=1)

    for xi, yi, sp in zip(x, y, species):
        plt.text(
            xi, yi,
            f"{sp}\n{yi:.3f}",
            ha="center", va="bottom",
            fontsize=8, rotation=45,
        )

    plt.axhline(float(threshold), linestyle="--", color="red")
    plt.title(f"Event Attention Gate Coverage Curve (Coverage >= {float(threshold):.2f})")
    plt.xlabel("Rank (sorted by Coverage desc)")
    plt.ylabel("Coverage")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=310)
    plt.close()


# =========================
# Gating Mechanism
# =========================
def gate_from_candidates(results_t, results_t_filtered, candidate_species, output_dir, coverage_threshold=0.50, enable_plot=True):
    if results_t.empty:
        raise RuntimeError("Gate: results_t is empty.")
    if results_t_filtered.empty:
        raise RuntimeError("Gate: results_t_filtered is empty. Try lower confidence_threshold.")

    os.makedirs(output_dir, exist_ok=True)

    if not candidate_species:
        gate_table = pd.DataFrame(columns=["Species", "N_full_actual", "N_acc_actual", "Coverage", "Coverage_pct", "Pass"])
        gate_selected = pd.DataFrame(columns=["Species", "N_full_actual", "N_acc_actual", "Coverage", "Coverage_pct", "Pass"])
        allow_list = pd.DataFrame({"Species": []})
        gate_table.to_csv(os.path.join(output_dir, "gate_table.csv"), index=False)
        gate_selected.to_csv(os.path.join(output_dir, "gate_selected.csv"), index=False)
        allow_list.to_csv(os.path.join(output_dir, "allow_list.csv"), index=False)
        return allow_list

    full = results_t.copy()
    full["Actual"] = full["Actual"].astype(str)

    acc = results_t_filtered.copy()
    acc["Actual"] = acc["Actual"].astype(str)

    full_counts = full["Actual"].value_counts().to_dict()
    acc_counts = acc["Actual"].value_counts().to_dict()

    rows = []
    for s in candidate_species:
        s = str(s)
        n_full = int(full_counts.get(s, 0))
        n_acc = int(acc_counts.get(s, 0))
        cov = (n_acc / n_full) if n_full > 0 else 0.0
        rows.append({"Species": s, "N_full_actual": n_full, "N_acc_actual": n_acc, "Coverage": cov})

    gate_table = (
        pd.DataFrame(rows)
        .sort_values(by=["Coverage", "N_full_actual"], ascending=[False, False])
        .reset_index(drop=True)
    )
    gate_table["Coverage_pct"] = (gate_table["Coverage"] * 100.0).round(2)
    gate_table["Pass"] = gate_table["Coverage"] >= float(coverage_threshold)

    gate_selected = gate_table[gate_table["Coverage"] >= float(coverage_threshold)].copy().reset_index(drop=True)
    allow_list = pd.DataFrame({"Species": gate_selected["Species"].astype(str).tolist()})

    gate_table.to_csv(os.path.join(output_dir, "gate_table.csv"), index=False)
    gate_selected.to_csv(os.path.join(output_dir, "gate_selected.csv"), index=False)
    allow_list.to_csv(os.path.join(output_dir, "allow_list.csv"), index=False)

    if enable_plot:
        plot_gate_coverage_threshold(
            gate_table=gate_table,
            threshold=float(coverage_threshold),
            out_png=os.path.join(output_dir, "gate_coverage.png"),
        )
    else:
        print("Skip plotting gate_coverage.png because ENABLE_PLOT=False")

    return allow_list


# =========================
# Save Model Configuration
# =========================
def save_model_config(step_path: str, num_classes: int):
    config = {
        "num_classes": int(num_classes),
        "MCL_FEAT_DIM": MCL_FEAT_DIM,
        "PROJ_DROPOUT": PROJ_DROPOUT,
        "ATTN_D_MODEL": ATTN_D_MODEL,
        "ATTN_NHEAD": ATTN_NHEAD,
        "ATTN_NUM_LAYERS": ATTN_NUM_LAYERS,
        "ATTN_DIM_FEEDFORWARD": ATTN_DIM_FEEDFORWARD,
        "ATTN_DROPOUT": ATTN_DROPOUT,
        "MAX_IMAGES_PER_EVENT": MAX_IMAGES_PER_EVENT,
        "ONLY_EVENTS_WITH_MULTI_IMAGES": ONLY_EVENTS_WITH_MULTI_IMAGES,
    }
    with open(os.path.join(step_path, "model_config.json"), "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# =========================
# main
# =========================
def main():
    img_path = "/data/users/ljl/datasets/SS/"
    step = 1

    step_path = create_step_folder(step)
    label_map_path = os.path.join(step_path, "label_to_index.json")

    train_csv = ""
    val_csv = ""

    num_classes = pd.read_csv(train_csv)["label"].astype(str).nunique()

    print("="*88 + f"\nconfidence_threshold:{confidence_threshold}\nRecall_Precision:{Recall_Precision}"+f"\ncoverage_threshold:{coverage_threshold}")
    print("="*88 + f"\n▶ ▶ ▶  EVENT ATTENTION DATASET  ◀ ◀ ◀\n\nNUM_CLASSES : {num_classes}\nTRAIN CSV   : {train_csv}\nVAL   CSV   : {val_csv}\n" + "="*88)

    train_data = CreateEventDatasetFromImages(
        csv_path=train_csv,
        file_path=img_path,
        transform=data_transforms["train"],
        label_map_path=label_map_path,
        save_map=True,
        training=True,
        max_images_per_event=MAX_IMAGES_PER_EVENT,
        only_events_with_multi_images=ONLY_EVENTS_WITH_MULTI_IMAGES,
    )

    val_data = CreateEventDatasetFromImages(
        csv_path=val_csv,
        file_path=img_path,
        transform=data_transforms["test"],
        label_map_path=label_map_path,
        save_map=False,
        training=False,
        max_images_per_event=MAX_IMAGES_PER_EVENT,
        only_events_with_multi_images=False,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        collate_fn=event_collate_fn,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=event_collate_fn,
    )

    model_path = os.path.join(step_path, "model.pth")
    save_model_config(step_path, num_classes)

    model, temperature = train_model(train_loader, val_loader, model_path, num_classes, patience=7)

    results_csv = os.path.join(step_path, "results.csv")
    results_t_csv = os.path.join(step_path, "results_t.csv")

    results, results_t = predict_val_data(
        model=model,
        val_loader=val_loader,
        label_map_path=label_map_path,
        temperature=temperature,
    )

    results.to_csv(results_csv, index=False)
    results_t.to_csv(results_t_csv, index=False)

    filtered_csv = os.path.join(step_path, "results_t_filtered.csv")
    results_t_filtered = filter_confidence(
        data_path=results_t_csv,
        output_path=filtered_csv,
        threshold=confidence_threshold,
    )

    metrics_path = os.path.join(step_path, "metrics.csv")
    calculate_metrics(filtered_csv, metrics_path)
    df_metrics = pd.read_csv(metrics_path)

    candidate_list_path = os.path.join(step_path, "candidate_list.csv")
    if df_metrics.empty:
        qualified_species = []
    else:
        qualified_species = df_metrics[
            (df_metrics["Recall"] >= Recall_Precision) &
            (df_metrics["Precision"] >= Recall_Precision)
        ]["Species"].astype(str).tolist()

    pd.DataFrame({"Species": sorted(qualified_species)}).to_csv(candidate_list_path, index=False)

    if qualified_species:
        print("="*88 + f"\nCANDIDATE SPECIES (n={len(qualified_species)})\n" + ", ".join(sorted(qualified_species)) + "\n" + "="*88)
    else:
        print(f"\n{'='*88}\n❌  CANDIDATE SPECIES: NONE  |  RP > {Recall_Precision}\nPATH: {candidate_list_path}\n{'='*88}\n")

    gate_path = os.path.join(step_path, "gate")
    os.makedirs(gate_path, exist_ok=True)

    df_allow = gate_from_candidates(
        results_t=results_t,
        results_t_filtered=results_t_filtered,
        candidate_species=qualified_species,
        output_dir=gate_path,
        coverage_threshold=coverage_threshold,
        enable_plot=ENABLE_PLOT,
    )

    print(f"\n{'='*88}\n✅ EVENT ATTENTION GATE FINAL SPECIES (n={len(df_allow)}):\n  " + (", ".join(df_allow['Species'].astype(str)) or "NONE") + f"\n{'='*88}")

    integrated_path = "./integrated_f1_scores.csv"
    df_allow.to_csv(integrated_path, index=False)


if __name__ == "__main__":
    main()
