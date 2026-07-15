import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax
from torchvision.models import resnext50_32x4d


# =========================
# Global Configuration
# =========================
batch_size = 40
num_worker = 6
GPU = "cuda:" + "0"
device = torch.device(GPU if torch.cuda.is_available() else "cpu")
print("device", device)

confidence_threshold = 0.95
MAX_IMAGES_PER_EVENT = None

data_transforms = {
    "test": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}


# =========================
# Event Dataset
# =========================
class CreateEventDatasetFromImages(Dataset):
    def __init__(
        self,
        csv_path,
        file_path,
        transform=None,
        label_map_path=None,
        max_images_per_event=None,
    ):
        self.file_path = file_path
        self.data_info = pd.read_csv(csv_path)

        required_cols = ["filename", "label", "event_id"]
        for c in required_cols:
            if c not in self.data_info.columns:
                raise RuntimeError(f"{csv_path} Missing required column: {c}. Event Attention testing requires a CSV file containing event_id.")

        self.data_info["filename"] = self.data_info["filename"].astype(str)
        self.data_info["label"] = self.data_info["label"].astype(str)
        self.data_info["event_id"] = self.data_info["event_id"].astype(str)

        self.transform = transform
        self.max_images_per_event = max_images_per_event

        if not (label_map_path and os.path.exists(label_map_path)):
            raise RuntimeError(f"label_to_index.json not found: {label_map_path}")

        with open(label_map_path, "r") as f:
            self.label_to_index = json.load(f)

        self.known_label_set = set(self.label_to_index.keys())

        missing_labels = sorted(set(self.data_info["label"].astype(str)) - self.known_label_set)
        self.missing_labels = missing_labels

        if len(missing_labels) > 0:
            print(
                "[WARNING] The test CSV contains species that were not seen during training. "
                "These species will not be added to the model's output classes and will only be retained in Actual: "
            )
            print("\n".join(missing_labels[:50]))

        self.groups = []
        for event_id, g in self.data_info.groupby("event_id", sort=False):
            self.groups.append((event_id, g.reset_index(drop=True)))

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
            g = g.iloc[: self.max_images_per_event].reset_index(drop=True)

        imgs, labels, filenames, actual_labels = [], [], [], []

        for _, row in g.iterrows():
            single_image_name = str(row["filename"])
            single_image_path = os.path.join(self.file_path, single_image_name)
            img_as_img = Image.open(single_image_path).convert("RGB")

            if self.transform:
                img_as_img = self.transform(img_as_img)

            label_str = str(row["label"])

            if label_str in self.label_to_index:
                label_index = int(self.label_to_index[label_str])
            else:
                label_index = -100

            imgs.append(img_as_img)
            labels.append(label_index)
            filenames.append(single_image_name)
            actual_labels.append(label_str)

        imgs = torch.stack(imgs, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        return imgs, labels, filenames, actual_labels, event_id

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

    all_filenames, all_actual_labels, all_event_ids = [], [], []

    for b, (imgs, labs, filenames, actual_labels, event_id) in enumerate(batch):
        n = imgs.shape[0]
        images[b, :n] = imgs
        labels[b, :n] = labs
        mask[b, :n] = True

        all_filenames.append(filenames)
        all_actual_labels.append(actual_labels)
        all_event_ids.append(event_id)

    return images, labels, mask, all_filenames, all_actual_labels, all_event_ids


# =========================
# Load the Mapping File
# =========================
def load_index_to_label(label_map_path: str):
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    return {int(v): k for k, v in label_map.items()}


# =========================
# load model config
# =========================
def load_model_config(step_folder: str):
    config_path = os.path.join(step_folder, "model_config.json")
    config = {
        "MCL_FEAT_DIM": 128,
        "PROJ_DROPOUT": 0.3,
        "ATTN_D_MODEL": 2048,
        "ATTN_NHEAD": 8,
        "ATTN_NUM_LAYERS": 2,
        "ATTN_DIM_FEEDFORWARD": 4096,
        "ATTN_DROPOUT": 0.1,
    }

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            obj = json.load(f)
        config.update(obj)
    else:
        print(f"[WARNING] model_config.json not found: {config_path}, use default config.")

    return config


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

        backbone = resnext50_32x4d(weights=None)
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

        self.proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, feat_dim),
        )

    def forward(self, x, mask, return_features: bool = False):
        B, N, C, H, W = x.shape

        x_valid = x[mask]
        feat_valid = self.backbone(x_valid)

        feat = x.new_zeros(B, N, self.in_features)
        feat[mask] = feat_valid

        padding_mask = ~mask.bool()
        context_feat = self.event_attention(
            feat,
            src_key_padding_mask=padding_mask,
        )

        logits = self.classifier(context_feat)

        if return_features:
            context_valid = context_feat[mask]
            z = self.proj(context_valid)
            z = F.normalize(z, dim=1)
            return logits, z

        return logits


# =========================
# load model
# =========================
def load_model(step_folder: str, num_classes: int, device: torch.device):
    model_path = os.path.join(step_folder, "model.pth")
    if not os.path.exists(model_path):
        raise RuntimeError(f"model.pth not found: {model_path}")

    config = load_model_config(step_folder)

    model = ResNeXt50EventAttentionContrastive(
        num_classes=num_classes,
        feat_dim=int(config["MCL_FEAT_DIM"]),
        dropout=float(config["PROJ_DROPOUT"]),
        attn_d_model=int(config["ATTN_D_MODEL"]),
        attn_nhead=int(config["ATTN_NHEAD"]),
        attn_num_layers=int(config["ATTN_NUM_LAYERS"]),
        attn_dim_feedforward=int(config["ATTN_DIM_FEEDFORWARD"]),
        attn_dropout=float(config["ATTN_DROPOUT"]),
    )

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()
    return model


# =========================
# load temperature
# =========================
def load_trained_temperature(step_folder: str) -> float:
    temp_path = os.path.join(step_folder, "model_T.json")
    if not os.path.exists(temp_path):
        raise RuntimeError(f"Temperature file not found: {temp_path}")

    with open(temp_path, "r") as f:
        obj = json.load(f)

    T = float(obj["temperature"])
    if not np.isfinite(T) or T <= 0:
        raise RuntimeError(f"Invalid temperature in {temp_path}: {T}")

    print(f"[LOAD] Temperature -> {T:.6f}")
    return T


# =========================
# load whitelist
# =========================
def load_allowlist(allowlist_csv: str) -> set:
    df = pd.read_csv(allowlist_csv)
    if "Species" not in df.columns:
        raise RuntimeError(f"{allowlist_csv} must contain column 'Species'")

    wl = (
        df["Species"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    return set(wl)


# =========================
# Predicted data
# =========================
def predict_test_data(model, test_loader, results_csv, label_map_path, species_above_threshold, temperature, confidence_threshold,):

    model.to(device)
    model.eval()

    with open(label_map_path, "r") as f:
        index_to_label = {int(v): k for k, v in json.load(f).items()}

    rows = []

    with torch.no_grad():
        val_bar = tqdm(test_loader, desc="Evaluating (scaled)")
        for inputs, labels, mask, all_filenames, all_actual_labels, all_event_ids in val_bar:
            inputs = inputs.to(device)
            mask = mask.to(device)

            logits = model(inputs, mask)
            outputs = logits / float(temperature)
            probabilities = softmax(outputs, dim=-1)
            predicted = probabilities.argmax(dim=-1)

            B, N, num_classes = probabilities.shape

            for b in range(B):
                for n in range(N):
                    if not bool(mask[b, n].item()):
                        continue

                    filename = all_filenames[b][n]
                    event_id = all_event_ids[b]
                    actual = all_actual_labels[b][n]
                    pred_label = index_to_label[int(predicted[b, n].item())]

                    row = {
                        "filename": filename,
                        "event_id": event_id,
                        "Actual": actual,
                        "Predicted": pred_label,
                    }

                    for c in range(num_classes):
                        row[f"Prob_{c}"] = float(probabilities[b, n, c].item())

                    rows.append(row)

    result = pd.DataFrame(rows)

    prob_cols = sorted(
        [c for c in result.columns if c.startswith("Prob_")],
        key=lambda x: int(x.split("_")[1]),
    )
    result = result[["filename", "event_id", "Actual", "Predicted"] + prob_cols]

    os.makedirs("raw", exist_ok=True)
    result.to_csv(os.path.join("raw", "test_all_scaled.csv"), index=False)

    result[prob_cols] = result[prob_cols].astype(float)
    max_prob = result[prob_cols].max(axis=1)

    mask_above = (
        result["Predicted"].isin(species_above_threshold)
        & (max_prob >= float(confidence_threshold))
    )

    df_results = result.loc[mask_above].copy()
    df_low = result.loc[~mask_above].copy()

    df_results[["filename", "Actual", "Predicted"]].to_csv(results_csv, index=False)
    df_low[["filename", "Actual", "Predicted"]].to_csv("low_config.csv", index=False)

    return result, df_results, df_low


# =========================
# main
# =========================
def main():
    img_path = ""
    test_csv = ""

    step_folder = "Step1"

    print("="*66 + f"\nNow is Processing Event Attention test csv...\nconfidence_threshold:{confidence_threshold}\nTEST CSV: {test_csv}\n" + "="*66)

    label_map_path = os.path.join(step_folder, "label_to_index.json")
    index_to_label = load_index_to_label(label_map_path)
    if not index_to_label:
        raise RuntimeError(f"label_to_index.json empty: {label_map_path}")

    allowlist_csv = "integrated_f1_scores.csv"
    allowlist = load_allowlist(allowlist_csv)

    print(f"Allow list species (n={len(allowlist)})\n" + "-"*66 + "\n" + "\n".join(f"- {s}" for s in sorted(allowlist)) + "\n" + "-"*66)

    test_data = CreateEventDatasetFromImages(
        csv_path=test_csv,
        file_path=img_path,
        transform=data_transforms["test"],
        label_map_path=label_map_path,
        max_images_per_event=MAX_IMAGES_PER_EVENT,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=event_collate_fn,
    )

    num_classes = len(index_to_label)
    model = load_model(step_folder, num_classes, device)

    T = load_trained_temperature(step_folder)

    results_csv = "Results.csv"

    result, df_results, df_low = predict_test_data(
        model=model,
        test_loader=test_loader,
        results_csv=results_csv,
        label_map_path=label_map_path,
        species_above_threshold=allowlist,
        temperature=T,
        confidence_threshold=confidence_threshold,
    )

    total_count = len(result)
    auto_count = len(df_results)
    print(f"auto_Rare: {auto_count / total_count * 100:.2f}%")


if __name__ == "__main__":
    main()
