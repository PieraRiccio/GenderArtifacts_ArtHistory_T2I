#!/usr/bin/env python3
"""
Fine-tune a ResNet18 binary classifier (male=0, female=1) on two folders of images.

Usage:
    python finetune_gender_classifier.py \
        --gpu 0 \
        --male-dir /path/to/male_images \
        --female-dir /path/to/female_images

Optional flags let you control split ratios, epochs, batch size, learning rate,
image size, and where the trained model / logs are saved. Run with --help for
the full list.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from collections import Counter

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class BinaryImageDataset(Dataset):
    """Loads (path, label) pairs and applies a transform on __getitem__."""

    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in IMG_EXTENSIONS and p.is_file()
    )


def build_samples(male_dir, female_dir):
    male_paths = list_images(male_dir)
    female_paths = list_images(female_dir)

    if len(male_paths) == 0:
        raise ValueError(f"No images found in male folder: {male_dir}")
    if len(female_paths) == 0:
        raise ValueError(f"No images found in female folder: {female_dir}")

    samples = [(str(p), 0) for p in male_paths] + [(str(p), 1) for p in female_paths]
    print(f"Found {len(male_paths)} male (label 0) and {len(female_paths)} "
          f"female (label 1) images -> {len(samples)} total")
    return samples


def split_samples(samples, train_frac, val_frac, seed):
    """Stratified split so train/val/test each keep the same class ratio."""
    by_class = {0: [], 1: []}
    for s in samples:
        by_class[s[1]].append(s)

    rng = random.Random(seed)
    train, val, test = [], [], []

    for label, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def class_counts(samples):
    counts = Counter(label for _, label in samples)
    return counts.get(0, 0), counts.get(1, 0)


def compute_class_weights(samples):
    """Inverse-frequency weights per class, normalized so they average to 1."""
    n0, n1 = class_counts(samples)
    total = n0 + n1
    n_classes = 2
    w0 = total / (n_classes * max(n0, 1))
    w1 = total / (n_classes * max(n1, 1))
    return torch.tensor([w0, w1], dtype=torch.float32)


def build_weighted_sampler(samples):
    """Per-sample weights so each epoch draws roughly balanced batches."""
    n0, n1 = class_counts(samples)
    weight_per_class = {0: 1.0 / max(n0, 1), 1: 1.0 / max(n1, 1)}
    sample_weights = [weight_per_class[label] for _, label in samples]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


def build_model(pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)  # 2-class output (male / female)
    return model


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(is_train)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    torch.set_grad_enabled(True)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, device, num_classes=2):
    """Runs inference over a loader and returns a dict of metrics:
    overall accuracy, balanced accuracy, per-class precision/recall/F1,
    and the confusion matrix.
    """
    model.eval()
    confusion = [[0] * num_classes for _ in range(num_classes)]  # confusion[true][pred]

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for t, p in zip(labels.tolist(), preds.tolist()):
                confusion[t][p] += 1

    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(num_classes))
    overall_acc = correct / total if total else 0.0

    per_class = {}
    recalls = []
    for c in range(num_classes):
        tp = confusion[c][c]
        support = sum(confusion[c])  # actual count of class c
        predicted_pos = sum(confusion[r][c] for r in range(num_classes))  # predicted as c

        recall = tp / support if support else 0.0      # per-class accuracy / sensitivity
        precision = tp / predicted_pos if predicted_pos else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)

        per_class[c] = {
            "support": support,
            "recall": recall,       # a.k.a. per-class accuracy
            "precision": precision,
            "f1": f1,
        }
        recalls.append(recall)

    balanced_acc = sum(recalls) / len(recalls) if recalls else 0.0

    return {
        "overall_acc": overall_acc,
        "balanced_acc": balanced_acc,
        "per_class": per_class,
        "confusion": confusion,
    }


def print_metrics(metrics, class_names=("male", "female")):
    print(f"Overall accuracy:  {metrics['overall_acc']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_acc']:.4f}  "
          f"(mean of per-class recall)")
    print(f"{'class':<8}{'support':>8}{'recall/acc':>12}{'precision':>11}{'f1':>8}")
    for c, name in enumerate(class_names):
        m = metrics["per_class"][c]
        print(f"{name:<8}{m['support']:>8}{m['recall']:>12.4f}"
              f"{m['precision']:>11.4f}{m['f1']:>8.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    header = "        " + "".join(f"{n:>10}" for n in class_names)
    print(header)
    for c, name in enumerate(class_names):
        row = "".join(f"{v:>10}" for v in metrics["confusion"][c])
        print(f"{name:<8}{row}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True,
                         help="GPU id to use (e.g. 0). Falls back to CPU if CUDA unavailable.")
    parser.add_argument("--male-dir", type=str, required=True,
                         help="Folder containing images labeled 0 (male).")
    parser.add_argument("--female-dir", type=str, required=True,
                         help="Folder containing images labeled 1 (female).")

    parser.add_argument("--train-frac", type=float, default=0.7,
                         help="Fraction of data used for training (default: 0.7).")
    parser.add_argument("--val-frac", type=float, default=0.15,
                         help="Fraction of data used for validation (default: 0.15). "
                              "The remainder is used for testing.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true",
                         help="If set, only the final classification layer is trained.")
    parser.add_argument("--class-weighting", type=str, default="none",
                         choices=["none", "loss", "sampler"],
                         help="How to handle class imbalance between the two folders: "
                              "'none' = no adjustment, 'loss' = inverse-frequency weighted "
                              "cross-entropy, 'sampler' = oversample the minority class each "
                              "epoch via WeightedRandomSampler (default: none).")
    parser.add_argument("--output", type=str, default="best_model.pt",
                         help="Path to save the best model checkpoint.")
    parser.add_argument("--model-selection", type=str, default="balanced_acc",
                         choices=["balanced_acc", "overall_acc"],
                         help="Metric used to pick the best checkpoint on the "
                              "validation set (default: balanced_acc, recommended "
                              "when classes are imbalanced).")

    args = parser.parse_args()

    if args.train_frac + args.val_frac >= 1.0:
        sys.exit("train-frac + val-frac must be < 1.0 so a non-empty test set remains.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---------------- Device ---------------- #
    if torch.cuda.is_available():
        if args.gpu >= torch.cuda.device_count():
            sys.exit(f"Requested GPU {args.gpu} but only "
                      f"{torch.cuda.device_count()} GPU(s) are visible.")
        device = torch.device(f"cuda:{args.gpu}")
    else:
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ---------------- Data ---------------- #
    samples = build_samples(args.male_dir, args.female_dir)
    train_samples, val_samples, test_samples = split_samples(
        samples, args.train_frac, args.val_frac, args.seed
    )
    print(f"Split -> train: {len(train_samples)}, val: {len(val_samples)}, "
          f"test: {len(test_samples)}")

    n0, n1 = class_counts(train_samples)
    ratio = max(n0, n1) / max(min(n0, n1), 1)
    print(f"Train class balance -> male(0): {n0}, female(1): {n1} (ratio {ratio:.2f}:1)")
    if ratio >= 1.5 and args.class_weighting == "none":
        print("WARNING: classes are noticeably imbalanced. Consider "
              "--class-weighting loss or --class-weighting sampler.")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = BinaryImageDataset(train_samples, transform=train_transform)
    val_ds = BinaryImageDataset(val_samples, transform=eval_transform)
    test_ds = BinaryImageDataset(test_samples, transform=eval_transform)

    if args.class_weighting == "sampler":
        sampler = build_weighted_sampler(train_samples)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                   num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ---------------- Model ---------------- #
    model = build_model(pretrained=True)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
        print("Backbone frozen: only training the final fc layer.")

    model = model.to(device)

    if args.class_weighting == "loss":
        weights = compute_class_weights(train_samples).to(device)
        print(f"Using class-weighted loss (male={weights[0]:.3f}, female={weights[1]:.3f})")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # ---------------- Train ---------------- #
    best_score = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)

        val_loss, _ = run_epoch(model, val_loader, device, criterion, optimizer=None)
        val_metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_overall_acc={val_metrics['overall_acc']:.4f} "
              f"val_balanced_acc={val_metrics['balanced_acc']:.4f}")

        score = val_metrics[args.model_selection]
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, args.output)
        print(f"\nSaved best model ({args.model_selection}={best_score:.4f}) to {args.output}")

    # ---------------- Test ---------------- #
    print(f"\nFinal evaluation on {len(test_samples)} held-out test images:")
    test_metrics = evaluate(model, test_loader, device)
    print_metrics(test_metrics)


if __name__ == "__main__":
    main()
