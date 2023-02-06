from typing import Any, Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader


def check_grad_cam(
    model: Any,
    dataloader: DataLoader,
    target_layer: List[Any],
    target_category: Any,
    dataloader_idx: Optional[int] = None,
    reshape_transform: Optional[Callable] = None,
    use_cuda: bool = False,
) -> Tuple[Any, ...]:
    grad_cam = GradCAMPlusPlus(
        model=model,
        target_layer=target_layer,
        use_cuda=use_cuda,
        reshape_transform=reshape_transform,
    )

    batch = next(iter(dataloader))
    images, labels = batch["image"], batch["label"]
    grad_cam.batch_size = len(images)
    logits = model.forward(images)
    if dataloader_idx:
        logits = logits[dataloader_idx]
    logits = logits.squeeze(1)
    pred = logits.sigmoid().detach().cpu().numpy() * 100
    labels = labels.cpu().numpy()

    grayscale_cam = grad_cam(
        input_tensor=images,
        target_category=target_category,
        eigen_smooth=True,
    )
    images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    images -= images.min()
    images /= images.max()
    return images, grayscale_cam, pred, labels


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
