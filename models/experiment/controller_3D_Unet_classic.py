import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.experiment.classic_3D_Unet import UNet3D
import os
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import random
from torch.utils.data import random_split

class NiftiDataset(Dataset):
    def __init__(self, images_dir, masks_dir, patch_size=(64, 64, 64), augment=False):
        self.image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".nii.gz")])
        self.mask_files  = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith(".nii.gz")])
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Загрузка
        img = nib.load(self.image_files[idx]).get_fdata().astype(np.float32)
        mask = nib.load(self.mask_files[idx]).get_fdata().astype(np.int64)

        # Нормализация (z-score)
        img = (img - img.mean()) / (img.std() + 1e-8)

        # Превращаем в torch.Tensor
        img = torch.from_numpy(img).unsqueeze(0)  # [C=1, D, H, W]
        mask = torch.from_numpy(mask)             # [D, H, W]

        # Вырезаем случайный патч (если данные большие)
        if self.patch_size is not None:
            img, mask = self.random_crop(img, mask, self.patch_size)

        # Аугментации (перевороты)
        if self.augment and random.random() > 0.5:
            img = torch.flip(img, dims=[2])   # flip H
            mask = torch.flip(mask, dims=[1])
        if self.augment and random.random() > 0.5:
            img = torch.flip(img, dims=[3])   # flip W
            mask = torch.flip(mask, dims=[2])

        return img, mask

    def random_crop(self, img, mask, patch_size):
        _, D, H, W = img.shape
        pd, ph, pw = patch_size
        # случайный сдвиг
        d = random.randint(0, max(0, D - pd))
        h = random.randint(0, max(0, H - ph))
        w = random.randint(0, max(0, W - pw))

        img_patch = img[:, d:d+pd, h:h+ph, w:w+pw]
        mask_patch = mask[d:d+pd, h:h+ph, w:w+pw]

        return img_patch, mask_patch


def _train_one_epoch(dataloader, model, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def _evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_loss = 0
    dice_score = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            # Dice (бинарный пример)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            intersection = ((preds == 1) & (labels == 1)).float().sum()
            union = (preds == 1).float().sum() + (labels == 1).float().sum()
            dice = (2. * intersection / (union + 1e-8))
            dice_score += dice.item()

    return epoch_loss / len(dataloader), dice_score / len(dataloader)


def conroller_3D_Unet_classic(cls_3DUnet_folder):
    # Пути к данным
    images_dir = os.path.join(cls_3DUnet_folder, "imagesTr")  # папка с .nii.gz изображениями
    masks_dir = os.path.join(cls_3DUnet_folder, "labelsTr")  # папка с масками

    dataset = NiftiDataset(images_dir, masks_dir, patch_size=(64, 64, 64), augment=True)

    # Разделим на train и val (например, 80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, out_channels=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = _train_one_epoch(train_loader, model, optimizer, criterion, device)
        val_loss, val_dice = _evaluate(val_loader, model, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs} "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f} "
              f"Val Dice: {val_dice:.4f}")

    torch.save(model.state_dict(), "unet3d.pth")
