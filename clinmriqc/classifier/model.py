"""
Artifact classifier inference — ResNet50 backbone, 7-class output.

Classes: clean, motion, noise, ghosting, bias_field, gibbs, zipper
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path

ARTIFACT_CLASSES = ['clean', 'motion', 'noise', 'ghosting', 'bias_field', 'gibbs', 'zipper']
N_CLASSES = len(ARTIFACT_CLASSES)
RESIZE_HW = 224


def build_model(arch: str = 'resnet50', n_classes: int = N_CLASSES) -> nn.Module:
    """Build a torchvision classification model with a replaced final layer.

    Supported architectures: resnet18, resnet34, resnet50, efficientnet_b0,
    efficientnet_b2, densenet121, mobilenet_v3_large, convnext_tiny.
    """
    if arch == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == 'resnet34':
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif arch == 'efficientnet_b2':
        m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif arch == 'densenet121':
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, n_classes)
    elif arch == 'mobilenet_v3_large':
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    elif arch == 'convnext_tiny':
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)
    else:
        raise ValueError(f'Unknown arch "{arch}"')
    return m


def load_model(checkpoint_path: str, arch: str = 'resnet50',
               device: str = None) -> nn.Module:
    """Load a trained checkpoint into a model and set it to eval mode.

    Device is auto-detected (CUDA if available, else CPU) when not specified.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(arch=arch)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                     weights_only=True))
    model.to(device).eval()
    return model


# Per-class thresholds tuned on real clinical data (bias_field and ghosting
# are common in unprocessed T1w scans so a higher bar reduces false positives).
_CLASS_THRESHOLDS = {
    'bias_field': 0.75,
    'ghosting':   0.60,
}


def predict_volume(
    model: nn.Module,
    volume: np.ndarray,
    threshold: float = 0.5,
    class_thresholds: dict = None,
    slice_step: int = 5,
    device: str = None,
) -> dict:
    """Return scan-level artifact probabilities via soft voting across axial slices.

    Samples slices from the central 20–80% of the volume (to avoid end-slices
    with little brain tissue), runs the classifier on each, and averages the
    softmax outputs. A class is flagged when its mean probability exceeds its
    threshold — either from _CLASS_THRESHOLDS or the caller-supplied
    class_thresholds dict (which takes precedence), falling back to `threshold`.
    """
    if device is None:
        device = next(model.parameters()).device

    effective_thresholds = {**_CLASS_THRESHOLDS, **(class_thresholds or {})}

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    n_slices = volume.shape[2]
    start = int(n_slices * 0.2)
    end   = int(n_slices * 0.8)

    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(start, end, slice_step):
            sl = volume[:, :, i].astype(np.float32)
            mn, mx = sl.min(), sl.max()
            if mx - mn < 1e-6:
                continue
            sl = (sl - mn) / (mx - mn)
            t = torch.tensor(sl[None]).repeat(3, 1, 1)
            t = transforms.functional.resize(t, [RESIZE_HW, RESIZE_HW],
                                             interpolation=transforms.InterpolationMode.BILINEAR)
            t = norm(t).unsqueeze(0).to(device)
            probs = torch.softmax(model(t), dim=1)[0].cpu().numpy()
            all_probs.append(probs)

    if not all_probs:
        return {'artifact_probabilities': {}, 'artifacts_detected': [], 'quality_passed': True}

    mean_probs = np.mean(all_probs, axis=0)
    artifact_probs = {name: round(float(p), 4) for name, p in zip(ARTIFACT_CLASSES, mean_probs)}
    detected = [name for name, p in artifact_probs.items()
                if name != 'clean' and p > effective_thresholds.get(name, threshold)]

    return {
        'artifact_probabilities': artifact_probs,
        'artifacts_detected':     detected,
        'quality_passed':         len(detected) == 0,
    }


def predict_slice(model: nn.Module, slice_arr: np.ndarray,
                  device: str = None) -> tuple[str, dict]:
    """Run inference on a single 2D slice (H, W), float32 normalised to [0,1]."""
    if device is None:
        device = next(model.parameters()).device
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    t = torch.tensor(slice_arr[None]).repeat(3, 1, 1)
    t = norm(t).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0].cpu().numpy()
    predicted = ARTIFACT_CLASSES[probs.argmax()]
    prob_dict = {name: round(float(p), 4) for name, p in zip(ARTIFACT_CLASSES, probs)}
    return predicted, prob_dict
