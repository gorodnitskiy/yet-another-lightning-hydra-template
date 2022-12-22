from typing import Any, Tuple

import matplotlib.pyplot as plt
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image


def reshape_transform(
    tensor: torch.Tensor, height: int = 7, width: int = 7
) -> torch.Tensor:
    """GradCam reshape_transform for ViT."""
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

    # like in CNNs.
    result = result.permute(0, 3, 1, 2)
    return result


def grad_cam_visualizer(
    grad_cam_outputs: Tuple[Any], figsize: Tuple[int, int] = (20, 20)
) -> None:
    """GradCam output visualizer."""
    plt.figure(figsize=figsize)
    for i, (image, grayscale_cam, pred, label) in enumerate(
        zip(grad_cam_outputs)
    ):
        plt.subplot(4, 4, i + 1)
        visualization = show_cam_on_image(image, grayscale_cam)

        plt.imshow(visualization)
        plt.title(f"pred:\n {pred}\n label: {label}")
        plt.axis("off")
