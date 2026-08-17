"""
Grad-CAM and Smoothgrad (saliency) utilities for the ResNet18 cup classifier.

Grad-CAM: highlights *regions* of the image that most influenced the prediction,
using gradients flowing into the last conv block (layer4).

Smoothgrad: highlights individual *pixels* the output is most
sensitive to, via d(output)/d(input).

Both are computed w.r.t. the raw logit (pre-sigmoid), which is standard practice
since sigmoid saturates and can shrink gradients near confident predictions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from PIL import Image


class GradCAM:
    """Grad-CAM for a single target conv layer (default: model.layer4)."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.layer4
        self.activations = None
        self.gradients = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(self, input_tensor):
        """
        input_tensor: (1, 3, H, W), requires no grad itself, but model params do.
        Returns: cam as a (H, W) numpy array normalized to [0, 1].
        """
        self.model.zero_grad()
        logit = self.model(input_tensor)  # (1, 1) raw logit
        logit.backward()

        # gradients/activations: (1, C, h, w)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # global-avg-pool grads -> (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, logit.item()


def compute_smoothgrad(model, input_tensor, n_samples: int = 30, noise_std: float = 0.15):
    """
    SmoothGrad saliency map: average |d(logit)/d(input)| over several noisy
    copies of the input, to cancel out the pixel-level noise that plain
    vanilla-gradient saliency maps suffer from.

    n_samples: number of noisy copies to average over (more = smoother, slower).
    noise_std: std-dev of the Gaussian noise added to the (normalized) input,
               as a fraction of its value range.

    Returns: saliency as a (H, W) numpy array normalized to [0, 1], and the
    logit from the *clean* (noise-free) input.
    """
    input_tensor = input_tensor.detach()

    with torch.no_grad():
        clean_logit = model(input_tensor).item()

    input_range = (input_tensor.max() - input_tensor.min()).item()
    sigma = noise_std * input_range

    accumulated_grad = torch.zeros_like(input_tensor)

    for _ in range(n_samples):
        noise = torch.randn_like(input_tensor) * sigma
        noisy_input = (input_tensor + noise).clone().detach().requires_grad_(True)

        model.zero_grad()
        logit = model(noisy_input)
        logit.backward()

        accumulated_grad += noisy_input.grad

    avg_grad = accumulated_grad / n_samples
    saliency = avg_grad.squeeze(0).abs().max(dim=0)[0]  # (H, W)
    saliency = saliency.cpu().numpy()
    if saliency.max() > 0:
        saliency = saliency / saliency.max()
    return saliency, clean_logit


def overlay_cam_on_image(pil_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Resize a (h, w) CAM to the image size, colorize with jet colormap, and blend."""
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(pil_image.size, resample=Image.BILINEAR)
    ) / 255.0

    heatmap = (cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)  # drop alpha channel
    heatmap_img = Image.fromarray(heatmap).convert("RGB")

    base = pil_image.convert("RGB").resize(pil_image.size)
    blended = Image.blend(base, heatmap_img, alpha=alpha)
    return blended


def saliency_to_image(pil_image: Image.Image, saliency: np.ndarray) -> Image.Image:
    """Resize a (h, w) saliency map to the image size and render as grayscale."""
    sal_resized = np.array(
        Image.fromarray((saliency * 255).astype(np.uint8)).resize(pil_image.size, resample=Image.BILINEAR)
    )
    return Image.fromarray(sal_resized).convert("L")
