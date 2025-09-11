import io

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def create_confmat_image(
    confusion_matrix: np.ndarray, weights: np.ndarray | None = None
) -> torch.Tensor:
    """Method to create a confusion matrix image from the confusion matrix.
    Args:
        confusion_matrix (np.ndarray): The confusion matrix to plot.
        weights (np.ndarray): The class counts to apply to each row in the confusion matrix.
    """
    fig, _ = create_confmat_figure(confusion_matrix, weights)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    image = torch.tensor(np.array(plt.imread(buf)).transpose(2, 0, 1))
    buf.close()
    return image


def create_confmat_figure(
    confusion_matrix: np.ndarray, weights: np.ndarray | None = None
) -> tuple[Figure, Axes]:
    """Method to create a confusion matrix image from the confusion matrix.
    Args:
        confusion_matrix (np.ndarray): The confusion matrix to plot.
        weights (np.ndarray): The class counts to apply to each row in the confusion matrix.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Total Classifications")

    if weights is not None:
        for row, weight in enumerate(weights):
            confusion_matrix[row, :] /= weight
        ax.set_title("Fraction of Classifications")

    sns.heatmap(confusion_matrix, annot=True, ax=ax, cmap="Blues", fmt="g")
    ax.set_xticklabels(["Wild", "Farmed"])
    ax.set_yticklabels(["Wild", "Farmed"])
    return fig, ax
