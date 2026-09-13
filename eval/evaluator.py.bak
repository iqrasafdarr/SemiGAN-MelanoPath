import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from eval.metrics import classification_metrics


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
    Evaluate the discriminator's cancer classifier on a patient-level split.
    """
    model.eval()

    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    y_true = []
    y_pred = []
    y_prob = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        class_logits, _ = model(images)

        probabilities = torch.softmax(
            class_logits,
            dim=1
        )

        predictions = probabilities.argmax(
            dim=1
        )

        y_true.extend(
            labels.cpu().numpy().tolist()
        )

        y_pred.extend(
            predictions.cpu().numpy().tolist()
        )

        y_prob.extend(
            probabilities[:, 1].cpu().numpy().tolist()
        )

    metrics = classification_metrics(
        y_true,
        y_pred,
        y_prob,
    )

    metrics["n_samples"] = len(y_true)

    return metrics


def save_metrics(metrics, output_path):
    """
    Save evaluation metrics as JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=2
        )
