# Cup Classifier
 
A lightweight binary image classifier that predicts whether an image contains
a cup, built with transfer learning on a pretrained ResNet18 backbone
(PyTorch). The project includes a small labelled
dataset, training with a two-phase fine-tuning strategy, running inference
with confidence scores, and visualising model decisions with
Grad-CAM and SmoothGrad.

This project accompanies a blog post I wrote [on learning representations](https://machinesight.github.io/),
which is about the foundations of learning, and useful representations that aids an AI algorithm to understand underlying patterns.

## Project structure
 
```
data/
├── train/
│   ├── cup/
│   └── not_cup/
└── val/
    ├── cup/
    └── not_cup/
├── test/                     # folder for sample test images
├── explanations/             # folder for Grad-CAM/SmoothGrad
├── train_cup_classifier.py   # Trains the binary classifier (transfer learning)
├── predict_model.py            # Runs inference; optional Grad-CAM / SmoothGrad
├── explain.py                 # Grad-CAM and SmoothGrad implementations
├── cup_classifier.pth        # ResNet18-based trained model
└── README.md
```
 
## Setup
 
```bash
pip install torch torchvision matplotlib pillow numpy
```

## Train
 
```bash
python train_cup_classifier.py
```
 
Training happens in two phases:
 
1. **Head-only**: the backbone is frozen, and only the new classification
   layer is trained.
2. **Fine-tune**: the last conv block (`layer4`) is unfrozen and fine-tuned
   at a low learning rate alongside the head.
The model outputs a single logit passed through a sigmoid, giving a direct
`P(cup)` confidence score. Weights are saved to `cup_classifier.pth`.
 
## Predict
 
```bash
python predict_model.py path/to/image.jpg    # could use one of the test samples
```
 
Output:
 
```
P(cup) = 0.9421
Prediction: CUP
```
 
## Explainability (Grad-CAM + SmoothGrad)
 
To visualize what the model is looking at:
 
```bash
python predict_model.py path/to/image.jpg --explain
```
 
This saves two images to `explanations/` (configurable via `--output-dir`):
 
- **`<image>_gradcam.png`** — a heatmap over image regions that most
  influenced the prediction, computed from gradients flowing into the last
  conv block.
- **`<image>_smoothgrad.png`** — a pixel-level sensitivity map, averaged over
  several noisy copies of the input to cancel out the noise that plain
  gradient-based saliency maps suffer from.

#### _Find the written article on [MachineSight](https://machinesight.github.io/)_
 
