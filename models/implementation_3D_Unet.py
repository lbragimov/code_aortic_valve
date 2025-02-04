import torch
import glob
import time
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import SimpleITK as sitk
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from data_preprocessing.log_worker import add_info_logging

global_loss_sum = [0, 0, 0, 0, 0]


def compute_per_channel_dice_3D(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
            input (torch.Tensor): NxCxSpatial input tensor
            target (torch.Tensor): NxCxSpatial target tensor
            epsilon (float): prevents division by zero
            weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    def flatten(tensor):
        """Flattens a given tensor such that the channel axis is first.
        The shapes are transformed as follows:
            (N, C, D, H, W) -> (C, N * D * H * W)
        """
        # number of channels
        C = tensor.size(1)
        # new axis order
        # Меняем порядок осей, чтобы каналы были первыми
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
        # Перестановка осей и преобразование в двумерный тензор
        transposed = tensor.permute(axis_order)
        # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
        return transposed.contiguous().view(C, -1)

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # Преобразование тензоров
    input = flatten(input) # Преобразуем предсказания
    target = flatten(target).float() # Преобразуем метки в float

    # compute per channel Dice Coefficient
    # Вычисляем пересечение (перекрытие) между input и target
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    # Вычисляем знаменатель: сумма квадратов input и target
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    # Возвращаем коэффициент Dice для каждого канала
    return 2 * (intersect / denominator.clamp(min=epsilon))


class DiceCoefficientMetric:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target):
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice_3D(input, target, epsilon=self.epsilon))


class DiceLoss(nn.Module):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    """

    def __init__(self):
        super().__init__()

    # def forward(self, input, target):
    #     # get probabilities from logits
    #     # normalization = nn.Sigmoid()
    #     # input = normalization(input)
    #
    #     # compute Dice score across all channels/classes
    #     input = torch.softmax(input, dim=1)  # Применяем Softmax перед вычислением Dice
    #     per_channel_dice = self.dice(input, target)
    #     # global_loss_sum[0] += float(per_channel_dice[0].detach().numpy())
    #     # global_loss_sum[1] += float(per_channel_dice[1].detach().numpy())
    #     # global_loss_sum[2] += float(per_channel_dice[2].detach().numpy())
    #     # global_loss_sum[3] += float(per_channel_dice[3].detach().numpy())
    #     # global_loss_sum[4] += 1
    #     return 1. - torch.mean(per_channel_dice)

    def forward(self, input, target):
        input = torch.softmax(input, dim=1)  # Softmax для логитов
        target_one_hot = F.one_hot(target, num_classes=input.shape[1])  # Преобразуем target в one-hot
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()  # Приводим в нужный формат

        per_channel_dice = compute_per_channel_dice_3D(input, target_one_hot)
        return 1. - torch.mean(per_channel_dice)

    def dice(self, input, target):
        return compute_per_channel_dice_3D(input, target)


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha=0.25, beta=0.75):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        # self.bce = nn.BCELoss()
        self.beta = beta
        self.dice = self.DiceLoss()
        t = 0

    def forward(self, input, target):
        return self.alpha * self.bce(input, target.double()) + self.beta * self.dice(input, target)


class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        add_info_logging(f"Shape of x: {x.shape}")
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        # Увеличение размера тензора x1
        x1 = self.up(x1)

        # Вычисление разницы по глубине, высоте и ширине
        diffZ = x2.size(2) - x1.size(2)  # Разница по глубине
        diffY = x2.size(3) - x1.size(3)  # Разница по высоте
        diffX = x2.size(4) - x1.size(4)  # Разница по ширине

        # Выравнивание x1 с учётом разницы размеров
        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,  # Ширина (X)
             diffY // 2, diffY - diffY // 2,  # Высота (Y)
             diffZ // 2, diffZ - diffZ // 2]  # Глубина (Z)
        )

        # Объединяем x1 и x2 по оси каналов
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, dropout=0.25, multiplier=8, bilinear=False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Базовый множитель каналов
        self.inc = DoubleConv3D(n_channels, 4 * multiplier)  # Входной блок
        self.down1 = Down3D(4 * multiplier, 8 * multiplier)  # Первый уровень
        self.down2 = Down3D(8 * multiplier, 16 * multiplier)  # Второй уровень
        self.down3 = Down3D(16 * multiplier, 32 * multiplier)  # Третий уровень

        factor = 2 if bilinear else 1
        self.down4 = Down3D(32 * multiplier, 64 * multiplier // factor)  # Четвёртый уровень

        self.up1 = Up3D(64 * multiplier, 32 * multiplier // factor, bilinear)  # Первый апскейлинг
        self.up2 = Up3D(32 * multiplier, 16 * multiplier // factor, bilinear)  # Второй апскейлинг
        self.up3 = Up3D(16 * multiplier, 8 * multiplier // factor, bilinear)  # Третий апскейлинг
        self.up4 = Up3D(8 * multiplier, 8 * multiplier, bilinear)  # Четвёртый апскейлинг

        self.outc = OutConv3D(8 * multiplier, n_classes)  # Финальный слой
        self.final_activation = nn.Softmax(dim=1)  # Для многоклассовой задачи

    def forward(self, input):
        # Прямое распространение через слои
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        logits = self.final_activation(x)
        return logits

    def use_checkpointing(self):
        # Поддержка контрольных точек для сохранения памяти
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNet3DTrainer:

    def __init__(self, n_classes=4, learning_rate=0.0001, weight_decay=0.01, epochs=300):
        self.model = UNet3D(n_channels=1, n_classes=n_classes).float()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # self.loss_criterion = DiceLoss()
        self.loss_criterion = nn.CrossEntropyLoss()
        self.eval_criterion = DiceCoefficientMetric()
        self.epochs = epochs

    def __move_to_device(self, input, device):
        if isinstance(input, tuple) or isinstance(input, list):
            return tuple([self.__move_to_device(x) for x in input])
        else:
            return torch.from_numpy(input).to(device)

    def __generate_result(self, input, mask):
        input_sitk = sitk.GetImageFromArray(input)
        # mask_united = np.zeros((mask.shape[1], mask.shape[2]))
        # for t in range(0, mask.shape[0]):
        #    mask_united[mask[t, :, :] >= 0.5] = t + 1
        # mask_united = mask[1, :, :] + mask[2, :, :] + mask[3, :, :] + mask[0, :, :]
        # Для объединения масок в 3D
        mask_united = mask.argmax(axis=0)  # Суммируем вдоль оси классов
        mask_sitk = sitk.GetImageFromArray(mask_united)
        return input_sitk, mask_sitk

    def train(self, train_dl, valid_dl, model_file):
        # Определяем устройство (GPU, если доступно)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)  # Перенос модели на устройство

        best_acc = 0.0

        start = time.time()
        train_loss, valid_loss = [], []

        for epoch in range(self.epochs):
            add_info_logging('Epoch {}/{}'.format(epoch, self.epochs - 1))
            add_info_logging('-' * 10)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model.train()  # Включаем режим обучения
                    dataloader = train_dl
                else:
                    self.model.eval()  # Включаем режим валидации
                    dataloader = valid_dl

                running_loss = 0.0
                running_acc = 0.0

                for step, (x, y) in enumerate(dataloader):
                    x = x.float()
                    y = y.float()
                    # Перенос данных на устройство
                    add_info_logging(f"Shape of x: {x.shape}")
                    x = x.to(device)
                    y = y.to(device)

                    # Forward pass
                    if phase == 'train':
                        self.optimizer.zero_grad()  # Обнуляем градиенты
                        outputs = self.model(x)  # Предсказание
                        loss = self.loss_criterion(outputs, y)  # Вычисление потерь
                        loss.backward()  # Backward pass
                        self.optimizer.step()  # Обновление параметров
                    else:
                        with torch.no_grad():  # Отключаем вычисление градиентов
                            outputs = self.model(x)
                            loss = self.loss_criterion(outputs, y)

                    # Вычисление метрик
                    acc = self.eval_criterion(outputs, y)
                    running_loss += loss.item() * x.size(0)  # Сумма потерь за батч
                    running_acc += acc.item() * x.size(0)  # Сумма точности за батч

                    if step % 5 == 0:
                        add_info_logging(f'Step {step}: Loss = {loss.item():.4f}, Acc = {acc.item():.4f}')

                # Усреднение статистик за эпоху
                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_acc / len(dataloader.dataset)

                add_info_logging(f'{phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

                # Сохранение потерь
                if phase == 'train':
                    train_loss.append(epoch_loss)
                else:
                    valid_loss.append(epoch_loss)

                # Сохранение лучшей модели
                # if phase == 'valid' and epoch_acc > max(valid_loss, default=0):
                #     torch.save(self.model.state_dict(), model_file)
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(self.model.state_dict(), model_file)

        time_elapsed = time.time() - start
        add_info_logging('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        torch.save(self.model.state_dict(), model_file + '_final.pth')

        return train_loss, valid_loss

    def test(self, test_dl, case_names, model_file, results_folder):
        # Определяем устройство (GPU, если доступно)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(model_file, map_location=device))  # Загружаем обученную модель
        self.model.to(device)  # Перенос модели на устройство
        self.model.eval()  # Перевод модели в режим тестирования

        start = time.time()  # Засекаем время выполнения

        global_count = 0  # Счётчик обработанных случаев

        with torch.no_grad():  # Отключаем вычисление градиентов
            for x, y in test_dl:
                x = x.float()
                y = y.float()
                # Перенос данных на устройство
                x = x.to(device)
                y = y.to(device)

                # Предсказание модели
                outputs = self.model(x)
                outputs_np = outputs.cpu().numpy()  # Перевод предсказаний в NumPy
                x_np = x.cpu().numpy()  # Перевод входных данных в NumPy
                y_np = y.cpu().numpy()  # Перевод истинных меток в NumPy

                # Обработка каждого элемента в батче
                for t in range(outputs_np.shape[0]):  # Цикл по примерам в батче
                    # Преобразование в формат SimpleITK
                    input_sitk, mask_sitk = self.__generate_result(x_np[t, 0, :, :, :], outputs_np[t, 0, :, :, :])

                    # Сохранение входного изображения
                    input_path = f"{results_folder}/im_{case_names[global_count]}.nii.gz"
                    sitk.WriteImage(input_sitk, input_path)

                    # Сохранение предсказанной маски
                    mask_path = f"{results_folder}/mask_{case_names[global_count]}.nii.gz"
                    sitk.WriteImage(mask_sitk, mask_path)

                    print(f"Saved results for case: {case_names[global_count]}")  # Логирование

                    global_count += 1  # Увеличиваем счётчик

        time_elapsed = time.time() - start  # Засекаем время
        print(f"Testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    def test_framework(self, slices, model_file):

        self.model.load_state_dict(torch.load(model_file))
        # self.model.eval()
        # self.model.cuda()

        global_count = 0
        slices_t = torch.tensor(slices.astype(float))
        slices_t = torch.unsqueeze(slices_t, 1)
        with torch.no_grad():
            output = self.model(slices_t).numpy()
            # results.append(output.numpy())

        # plt.imshow(output[2, 0, :, :])
        # plt.show()

        # plt.imshow(output[2, 1, :, :])
        # plt.show()

        # plt.imshow(output[2, 2, :, :])
        # plt.show()

        # plt.imshow(output[2, 3, :, :])
        # plt.show()
        return output


class WrapperUnet:

    @staticmethod
    def try_unet3d_training(folder):
        loader = DataloaderSeg3D(folder + '/data')
        loader.generate_data_loaders(0.15, 2)

        trainer = UNet3DTrainer()
        trainer.train(loader.train_dl, loader.valid_dl, folder + '/models/model_weights.pth')

    @staticmethod
    def try_unet3d_testing(folder):
        loader = DataloaderSeg3D(folder + '/test_data')
        loader.generate_data_loaders(0, 2)

        trainer = UNet3DTrainer()
        trainer.test(loader.test_dl, loader.case_names,
                     folder + '/models/model_weights.pth', folder + '/results')


class DataloaderSeg3D:

    def __init__(self, data_folder):
        subfolders = glob.glob(data_folder + '/*')
        self.database = DatabaseImSegNII(subfolders)
        self.case_names = []
        for subfolder in subfolders:
            self.case_names.append(Path(subfolder).parts[-1])

    def generate_data_loaders(self, valid_prop, batch_size, shuffle=True):
        if valid_prop > 0:
            valid_num = int(valid_prop * len(self.database))
            train_ds, valid_ds = torch.utils.data.random_split(self.database,
                                                               (len(self.database) - valid_num, valid_num))
            self.train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
            self.valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=shuffle)
        else:
            self.test_dl = DataLoader(self.database, batch_size=batch_size)


class DatabaseImSegNII(Dataset):

    def __init__(self, subfolders):
        self.images = []
        self.masks = []
        self.transform = ToTensor()
        for i in range(0, len(subfolders)):
            self.images.append(self._load_nii(Path(subfolders[i]) / "image.nii.gz"))
            self.masks.append(self._load_nii(Path(subfolders[i]) / "mask.nii.gz"))
        if len(self.images) > 0:
            self.normalize_images()

    def normalize_images(self):
        minF = np.amin(self.images[0])
        maxF = np.amax(self.images[0])
        for i in range(1, len(self.images)):
            minF = min(minF, np.amin(self.images[i]))
            maxF = max(maxF, np.amax(self.images[i])) # min(maxF, np.amax(self.images[0]))
        # minF = np.amin(np.array(self.images))
        # maxF = np.amax(np.array(self.images))

        for i in range(0, len(self.images)):
            self.images[i] = ((self.images[i] - minF)/(maxF - minF)).astype(np.float32)

    def _load_nii(self, file_path):
        """
        Загружает файл .nii.gz и возвращает данные в виде NumPy массива.
        """
        itk_image = sitk.ReadImage(file_path)  # Чтение файла с помощью SimpleITK
        image = sitk.GetArrayFromImage(itk_image)  # Преобразование в NumPy массив
        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # img = self.images[idx]
        # mask = self.masks[idx]
        #
        # # add_info_logging(f"🔹 Загружено изображение {idx}: img.shape={img.shape} mask.shape={mask.shape}")
        # return self.transform(self.images[idx]), torch.tensor(self.masks[idx], dtype=torch.long)
        img = self.images[idx]
        mask = self.masks[idx]
        add_info_logging(f"🔹 Загружено изображение {idx}: img.shape={img.shape} mask.shape={mask.shape}")

        # 🔹 Если изображение имеет 3 оси (D, H, W), добавляем ось канала
        if len(img.shape) == 3:
            img = img[np.newaxis, :, :, :]  # Теперь (1, D, H, W)

        img = torch.tensor(img, dtype=torch.float32)  # Преобразуем в PyTorch тензор
        mask = torch.tensor(mask, dtype=torch.long)  # Маска в long (для CrossEntropyLoss)

        return img, mask
