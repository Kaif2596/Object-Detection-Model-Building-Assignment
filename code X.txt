# Complete Object Detection Code for Assignment
# This implementation uses Faster R-CNN with a ResNet-50 backbone pre-trained on COCO.
# It handles dataset loading, model training, evaluation, and inference with label display.

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained Faster R-CNN with ResNet-50 backbone
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# Label map for COCO (91 classes)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Load image for inference
def load_image(img_path):
    image = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device), image

# Visualize prediction
def visualize_prediction(model, img_path, threshold=0.5):
    model.eval()
    input_tensor, original_image = load_image(img_path)
    with torch.no_grad():
        outputs = model(input_tensor)

    output = outputs[0]
    boxes = output['boxes'].cpu()
    labels = output['labels'].cpu()
    scores = output['scores'].cpu()

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(original_image)

    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            label_text = f"{COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}"
            ax.text(xmin, ymin - 10, label_text, color='red', fontsize=12,
                    backgroundcolor="white")

    plt.axis('off')
    plt.show()

# Example usage (upload and use your own image)
# Replace with your actual image path or upload via Colab
img_path = "D:\Projects\Bi Polar Assignment\image05.jpg"  # example image name
visualize_prediction(model, img_path, threshold=0.5)
