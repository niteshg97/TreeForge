"""Confusion matrix visualization utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.metrics import confusion_matrix


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    filename: str = "assets/images/confusion_matrix.png",
) -> str:
    """Save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Optional class names.
        filename: Output image path.

    Returns:
        Path of the saved image.
    """

    cm = confusion_matrix(y_true, y_pred)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm)

    plt.colorbar(image)

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

    plt.tight_layout()

    plt.savefig(output_path, dpi=300)

    plt.close(fig)

    return str(output_path)
