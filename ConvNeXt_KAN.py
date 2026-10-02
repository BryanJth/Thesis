import os
import csv
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.metrics import classification_report, confusion_matrix

from kan import KANLinear


BATCH_SIZE = 32
NUM_EPOCHS = 15
N_RUNS = 10
BASE_SEED = 3703

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
STEP_SIZE = 5
GAMMA = 0.5

HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 512
NUM_CLASSES = 10


DATA_ROOT = "./data"
OUTPUT_DIR = "runs_convnext_kan"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ConvNeXtKAN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT
        self.convnext = convnext_tiny(weights=weights)

        for param in self.convnext.parameters():
            param.requires_grad = False

        in_features = self.convnext.classifier[2].in_features
        self.convnext.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            KANLinear(in_features, HIDDEN_DIM_1),
            nn.BatchNorm1d(HIDDEN_DIM_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            KANLinear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.BatchNorm1d(HIDDEN_DIM_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            KANLinear(HIDDEN_DIM_2, num_classes),
        )

    def forward(self, x):
        x = self.convnext(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader, train_dataset.classes


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += batch_size

    epoch_loss = total_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, test_loader, criterion, device, class_names):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(test_loader.dataset)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    accuracy = report["accuracy"] * 100.0
    cm = confusion_matrix(all_labels, all_preds)
    return epoch_loss, accuracy, report, cm


def run_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, class_names = get_dataloaders()

    summary_path = os.path.join(OUTPUT_DIR, "summary_convnext_kan.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run",
            "seed",
            "final_test_accuracy",
            "best_test_accuracy",
            "macro_precision_final",
            "macro_recall_final",
            "weighted_precision_final",
            "weighted_recall_final",
        ])

    for run_idx in range(1, N_RUNS + 1):
        seed = BASE_SEED + run_idx
        set_seed(seed)

        run_dir = os.path.join(OUTPUT_DIR, f"run_{run_idx:02d}")
        os.makedirs(run_dir, exist_ok=True)

        model = ConvNeXtKAN(num_classes=NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=STEP_SIZE,
            gamma=GAMMA,
        )

        epoch_csv = os.path.join(run_dir, "epoch_metrics.csv")
        with open(epoch_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc"])

        best_acc = 0.0
        final_report = None
        final_cm = None
        final_acc = 0.0

        for epoch in range(1, NUM_EPOCHS + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc, report, cm = evaluate(model, test_loader, criterion, device, class_names)

            best_acc = max(best_acc, test_acc)
            final_report = report
            final_cm = cm
            final_acc = test_acc

            print(
                f"Run {run_idx:02d} | Epoch {epoch:02d}/{NUM_EPOCHS} | "
                f"LR {current_lr:.6g} | Train Acc {train_acc:.2f}% | Test Acc {test_acc:.2f}%"
            )

            with open(epoch_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, current_lr, train_loss, train_acc, test_loss, test_acc])

            scheduler.step()

        torch.save(model.state_dict(), os.path.join(run_dir, "convnext_kan_cifar10.pth"))
        np.savetxt(os.path.join(run_dir, "confusion_matrix.csv"), final_cm, fmt="%d", delimiter=",")

        with open(os.path.join(run_dir, "classification_report.txt"), "w") as f:
            f.write(classification_report(
                [], [], labels=[], target_names=[], zero_division=0
            ) if False else str(final_report))

        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                run_idx,
                seed,
                final_acc,
                best_acc,
                final_report["macro avg"]["precision"],
                final_report["macro avg"]["recall"],
                final_report["weighted avg"]["precision"],
                final_report["weighted avg"]["recall"],
            ])


if __name__ == "__main__":
    run_experiment()
