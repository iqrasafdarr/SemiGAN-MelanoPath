import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from eval.metrics import classification_metrics


class Evaluator:
    """
    Research evaluation utilities for classification, calibration,
    uncertainty-aware abstention, and checkpoint evaluation.
    """

    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def evaluate_classification(self, y_true, y_pred, y_prob):
        """
        Compute standard binary classification metrics.

        Parameters
        ----------
        y_true:
            Ground-truth class labels.
        y_pred:
            Predicted class labels.
        y_prob:
            Probability assigned to the positive class.
        """
        y_true = self._to_numpy(y_true).astype(int)
        y_pred = self._to_numpy(y_pred).astype(int)
        y_prob = self._to_numpy(y_prob).astype(float)

        return classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )

    def compute_ece(self, y_true, y_prob, n_bins=10):
        """
        Expected Calibration Error for binary classification.

        y_prob may contain either:
          - positive-class probabilities of shape [N]
          - two-class probabilities of shape [N, 2]
        """
        y_true = self._to_numpy(y_true).astype(int)
        y_prob = self._to_numpy(y_prob).astype(float)

        if y_prob.ndim == 2:
            if y_prob.shape[1] != 2:
                raise ValueError(
                    f"Expected binary class probabilities, got shape {y_prob.shape}"
                )

            confidence = np.max(y_prob, axis=1)
            predictions = np.argmax(y_prob, axis=1)

        elif y_prob.ndim == 1:
            positive_prob = np.clip(y_prob, 0.0, 1.0)

            predictions = (positive_prob >= 0.5).astype(int)

            confidence = np.where(
                predictions == 1,
                positive_prob,
                1.0 - positive_prob,
            )

        else:
            raise ValueError(
                f"Expected probability array with 1 or 2 dimensions, got {y_prob.ndim}"
            )

        if len(y_true) != len(predictions):
            raise ValueError("y_true and y_prob must have the same number of samples.")

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

        ece = 0.0

        for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
            if upper == 1.0:
                mask = (confidence >= lower) & (confidence <= upper)
            else:
                mask = (confidence >= lower) & (confidence < upper)

            if not np.any(mask):
                continue

            bin_accuracy = np.mean(predictions[mask] == y_true[mask])
            bin_confidence = np.mean(confidence[mask])
            bin_weight = np.mean(mask)

            ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return float(ece)

    def abstention_curve(
        self,
        y_true,
        y_prob,
        uncertainty,
        num_points=101,
    ):
        """
        Compute risk-coverage / abstention behaviour.

        Samples with the highest uncertainty are removed first.

        Returns
        -------
        coverage : np.ndarray
            Fraction of samples retained.

        accuracy : np.ndarray
            Classification accuracy among retained samples.
        """
        y_true = self._to_numpy(y_true).astype(int)
        y_prob = self._to_numpy(y_prob).astype(float)
        uncertainty = self._to_numpy(uncertainty).astype(float)

        if y_prob.ndim == 2:
            predictions = np.argmax(y_prob, axis=1)
        else:
            predictions = (y_prob >= 0.5).astype(int)

        uncertainty = uncertainty.reshape(-1)

        if not (
            len(y_true)
            == len(predictions)
            == len(uncertainty)
        ):
            raise ValueError(
                "y_true, y_prob and uncertainty must contain the same number of samples."
            )

        n_samples = len(y_true)

        if n_samples == 0:
            return np.array([]), np.array([])

        # Lowest uncertainty = most confident sample.
        order = np.argsort(uncertainty)

        coverage_values = np.linspace(
            1.0 / n_samples,
            1.0,
            min(num_points, n_samples),
        )

        retained_counts = np.unique(
            np.clip(
                np.round(coverage_values * n_samples).astype(int),
                1,
                n_samples,
            )
        )

        coverage = []
        accuracy = []

        for retained in retained_counts:
            selected = order[:retained]

            acc = np.mean(
                predictions[selected] == y_true[selected]
            )

            coverage.append(retained / n_samples)
            accuracy.append(acc)

        return (
            np.asarray(coverage, dtype=float),
            np.asarray(accuracy, dtype=float),
        )

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()

        return np.asarray(value)


@torch.no_grad()
def evaluate_classifier(
    model,
    dataset,
    indices,
    device,
    batch_size=32,
    num_workers=0,
):
    """
    Deterministic classifier evaluation on a dataset subset.
    """
    model.eval()

    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    y_true = []
    y_pred = []
    y_prob = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        class_logits, _ = model(images)

        probabilities = torch.softmax(class_logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        y_true.extend(
            labels.detach().cpu().numpy().tolist()
        )

        y_pred.extend(
            predictions.detach().cpu().numpy().tolist()
        )

        y_prob.extend(
            probabilities[:, 1].detach().cpu().numpy().tolist()
        )

    metrics = classification_metrics(
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(y_prob),
    )

    metrics["n_samples"] = len(y_true)

    return metrics


def save_metrics(metrics, output_path):
    """
    Save metric dictionary as JSON while safely converting NumPy values.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, (np.floating, np.integer)):
            return value.item()

        if isinstance(value, dict):
            return {
                key: convert(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]

        return value

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            convert(metrics),
            file,
            indent=2,
        )
