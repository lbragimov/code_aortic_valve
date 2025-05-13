import numpy as np
import SimpleITK as sitk
from sklearn.metrics import jaccard_score, f1_score
from typing import Literal, Tuple
from medpy.metric.binary import hd, assd


class landmarking_locked:
    """пока не используется"""

    def __init__(self, probability_maps):
        self.probability_maps = probability_maps

    def __prepare_probability_lists(self):
        # Step 1: Extract nonzero probability locations for each landmark
        nonzero_locs = {}  # Store voxel locations where probability > 0
        probs = {}  # Store corresponding probabilities

        for key, prob_map in self.probability_maps.items():
            indices = np.argwhere(prob_map > 0)  # Get all non-zero locations
            probabilities = prob_map[prob_map > 0]  # Get probability values

            # Normalize probabilities to sum to 1 (so they form a proper distribution)
            probabilities /= np.sum(probabilities)

            # Store valid locations and their probabilities
            nonzero_locs[key] = indices
            probs[key] = probabilities

        return nonzero_locs, probs

    def __measure_metrics(self, landmarks):
        pass

    def simulation_MC(self, num_simulations):
        # Step 2: Monte Carlo Sampling
        results = []  # Store weighted measurements

        nonzero_locs, probs = self.__prepare_probability_lists()

        for _ in range(num_simulations):
            sampled_landmarks = {}

            # Step 3: Randomly select a voxel location for each landmark
            for key in self.probability_maps.keys():
                sampled_index = np.random.choice(len(probs[key]), p=probs[key])  # Weighted random selection
                sampled_landmarks[key] = nonzero_locs[key][sampled_index]  # Get the selected voxel

            # Step 4: Compute measurement (Replace with actual function)
            measurement = self.__measure_metrics(sampled_landmarks)

            # Step 5: Store weighted measurement (measurement * probability product)
            weight = np.prod([probs[key][sampled_index] for key in self.probability_maps.keys()])
            results.append(measurement * weight)

        return results


def evaluate_segmentation(true_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int = 1,
                          average: Literal['macro', 'weighted'] = 'macro'):
    """
    Вычисляет Dice и IoU между масками.

    Parameters:
        true_mask (np.ndarray): Ground truth маска (2D или 3D).
        pred_mask (np.ndarray): Предсказанная маска (2D или 3D).
        num_classes (int): Количество классов. Если 1 — бинарная сегментация.
        average (str): Способ усреднения ('macro' или 'weighted').

    Returns:
        dict: {'Dice': ..., 'IoU': ...}
    """
    true_flat = true_mask.flatten()
    pred_flat = pred_mask.flatten()

    metrics = {}

    if num_classes == 1:
        dice = f1_score(true_flat, pred_flat)
        iou = jaccard_score(true_flat, pred_flat)

        metrics["Dice"] = dice
        metrics["IoU"] = iou

        try:
            # Преобразуем в бинарные маски
            true_bin = (true_mask > 0).astype(np.bool_)
            pred_bin = (pred_mask > 0).astype(np.bool_)

            if np.count_nonzero(true_bin) == 0 or np.count_nonzero(pred_bin) == 0:
                metrics["HD"] = np.nan
                metrics["ASSD"] = np.nan
            else:
                metrics["HD"] = hd(pred_bin, true_bin)
                metrics["ASSD"] = assd(pred_bin, true_bin)
        except Exception as e:
            metrics["HD"] = np.nan
            metrics["ASSD"] = np.nan
    else:
        dice = f1_score(true_flat, pred_flat, average=average, labels=range(num_classes))
        iou = jaccard_score(true_flat, pred_flat, average=average, labels=range(num_classes))
        metrics["Dice"] = dice
        metrics["IoU"] = iou
        metrics["HD"] = np.nan  # многоклассовую HD/ASSD сложнее интерпретировать
        metrics["ASSD"] = np.nan

    return metrics


def hausdorff_distance_sitk(mask1: np.ndarray, mask2: np.ndarray,
                            spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
    """
    Вычисляет Hausdorff и HD95 расстояния между двумя 3D масками с учетом физического размера вокселя.

    Parameters:
        mask1 (np.ndarray): Ground truth бинарная 3D маска.
        mask2 (np.ndarray): Предсказанная бинарная 3D маска.
        spacing (tuple): Физический размер вокселя (мм), по осям (z, y, x).

    Returns:
        dict: {'Hausdorff': ..., 'HD95': ...}
    """
    mask1_itk = sitk.GetImageFromArray(mask1.astype(np.uint8))
    mask2_itk = sitk.GetImageFromArray(mask2.astype(np.uint8))

    mask1_itk.SetSpacing(spacing)
    mask2_itk.SetSpacing(spacing)

    hd_filter = sitk.HausdorffDistanceImageFilter()
    hd_filter.Execute(mask1_itk, mask2_itk)

    return {
        'Hausdorff': hd_filter.GetHausdorffDistance(),
        'HD95': hd_filter.Get95PercentHausdorffDistance()
    }

