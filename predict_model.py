"""
Run inference with the trained cup classifier, with optional Grad-CAM visualization.

Usage:
    python predict_model.py path/to/image.jpg
    python predict_model.py path/to/image.jpg --explain
    python predict_model.py path/to/image.jpg --explain --output-dir explanations
    python predict_model.py path/to/image.jpg --weights cup_classifier.pth
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from explain import GradCAM, compute_smoothgrad, overlay_cam_on_image, saliency_to_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Kept separate from the normalized tensor so we have an unmodified image to
# overlay the CAM / saliency map onto.
resize_crop = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])
to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model(weights_path="cup_classifier.pth"):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def prepare_input(image_path):
    """Returns (display_image [224x224 PIL, unnormalized], input_tensor [1,3,224,224])."""
    img = Image.open(image_path).convert("RGB")
    display_img = resize_crop(img)  # what we'll overlay the heatmap on
    input_tensor = to_tensor_norm(display_img).unsqueeze(0).to(DEVICE)
    return display_img, input_tensor


def predict(model, input_tensor):
    with torch.no_grad():
        logit = model(input_tensor)
        prob_cup = torch.sigmoid(logit).item()
    return prob_cup


def explain(model, display_img, input_tensor, image_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    # --- Grad-CAM ---
    cam_extractor = GradCAM(model, target_layer=model.layer4)
    cam, _ = cam_extractor.generate(input_tensor)
    cam_extractor.remove_hooks()
    cam_overlay = overlay_cam_on_image(display_img, cam)
    cam_path = output_dir / f"{stem}_gradcam.png"
    cam_overlay.save(cam_path)

    # --- saliency smoothgrad ---
    saliency, _ = compute_smoothgrad(model, input_tensor)
    saliency_img = saliency_to_image(display_img, saliency)
    saliency_path = output_dir / f"{stem}_smoothgrad.png"
    saliency_img.save(saliency_path)

    return cam_path, saliency_path


def main():
    parser = argparse.ArgumentParser(description="Predict cup / not-cup, with optional Grad-CAM + saliency.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--weights", type=str, default="cup_classifier.pth",
                         help="Path to trained model weights")
    parser.add_argument("--explain", action="store_true",
                         help="Also compute and save Grad-CAM and gradient saliency images")
    parser.add_argument("--output-dir", type=str, default="explanations",
                         help="Directory to save Grad-CAM / saliency (smoothgrad) images (used with --explain)")
    args = parser.parse_args()

    model = load_model(args.weights)
    display_img, input_tensor = prepare_input(args.image_path)

    confidence = predict(model, input_tensor)
    print(f"P(cup) = {confidence:.4f}")
    print("Prediction:", "CUP" if confidence > 0.5 else "NOT CUP")

    if args.explain:
        cam_path, saliency_path = explain(model, display_img, input_tensor, args.image_path, args.output_dir)
        print(f"Grad-CAM saved to:   {cam_path}")
        print(f"Saliency saved to:   {saliency_path}")


if __name__ == "__main__":
    main()
