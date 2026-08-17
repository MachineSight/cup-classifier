"""
Binary "cup / not-cup" classifier using transfer learning (ResNet18).

Expected folder structure:

    data/
        train/
            cup/        *.jpg
            not_cup/    *.jpg
        val/
            cup/        *.jpg
            not_cup/    *.jpg
"""

import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR = Path("data")
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. Data pipeline

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
}

image_datasets = {
    split: datasets.ImageFolder(DATA_DIR / split, data_transforms[split])
    for split in ["train", "val"]
}

# ImageFolder assigns class indices alphabetically; confirm "cup" -> which index
print("Class-to-index mapping:", image_datasets["train"].class_to_idx)
# We want cup_idx to know which logit/prob corresponds to "cup"
CUP_IDX = image_datasets["train"].class_to_idx["cup"]

dataloaders = {
    split: DataLoader(
        image_datasets[split],
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=NUM_WORKERS,
    )
    for split in ["train", "val"]
}
dataset_sizes = {split: len(image_datasets[split]) for split in ["train", "val"]}

# ---------------------------------------------------------------------------
# 2. Model: pretrained ResNet18, single-logit output (binary via BCE)

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # single logit -> sigmoid = P(cup)
    return model.to(DEVICE)


model = build_model()
criterion = nn.BCEWithLogitsLoss()

# ---------------------------------------------------------------------------
# 3. Training loop (supports two phases: head-only, then fine-tune)

def train_model(model, optimizer, scheduler, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                # label 1.0 = cup, 0.0 = not_cup (map via CUP_IDX)
                labels = (labels == CUP_IDX).float().unsqueeze(1).to(DEVICE)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)  # raw logits
                    loss = criterion(outputs, labels)
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train" and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"  {phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Best val acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    start = time.time()

    # Phase 1: train only the new head, backbone frozen
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = train_model(model, optimizer, scheduler, num_epochs=5)

    # Phase 2: unfreeze last block + fc, fine-tune at a low LR
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    model = train_model(model, optimizer, scheduler, num_epochs=5)

    torch.save(model.state_dict(), "cup_classifier.pth")
    print(f"Done in {time.time() - start:.1f}s. Saved to cup_classifier.pth")
