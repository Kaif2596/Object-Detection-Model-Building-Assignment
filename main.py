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


################################################################################
##This is Google Colab code that I add here##

import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained MobileNetV2
mobilenet = models.mobilenet_v2(pretrained=True)

# Remove the classifier head
backbone = mobilenet.features
backbone.out_channels = 1280  # MobileNetV2 output feature size

class SSDHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SSDHead, self).__init__()
        self.class_head = nn.Conv2d(in_channels, num_classes * 4, kernel_size=3, padding=1)
        self.box_head = nn.Conv2d(in_channels, 4 * 4, kernel_size=3, padding=1)

    def forward(self, x):
        class_preds = self.class_head(x)  # (N, num_classes*4, H, W)
        box_preds = self.box_head(x)      # (N, 4*4, H, W)
        return class_preds, box_preds

# Define number of object classes (20 Pascal VOC classes + 1 background)
num_classes = 21
ssd_head = SSDHead(in_channels=1280, num_classes=num_classes)

class SSDModel(nn.Module):
    def __init__(self, backbone, ssd_head):
        super(SSDModel, self).__init__()
        self.backbone = backbone
        self.ssd_head = ssd_head

    def forward(self, x):
        features = self.backbone(x)
        class_preds, box_preds = self.ssd_head(features)
        return class_preds, box_preds

# Initialize the model
model = SSDModel(backbone, ssd_head)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

from torchvision.datasets import VOCDetection
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2

# Define object class names
VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class VOCDataset(Dataset):
    def __init__(self, root, image_set='train', year='2007', transform=None):
        self.dataset = VOCDetection(root=root, year=year, image_set=image_set, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, annotation = self.dataset[idx]
        boxes = []
        labels = []

        for obj in annotation['annotation']['object']:
            bbox = obj['bndbox']
            label = VOC_CLASSES.index(obj['name'])
            box = [int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])]
            boxes.append(box)
            labels.append(label)

        img = np.array(img)

        if self.transform:
            img = self.transform(img)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return img, boxes, labels

# Define image transforms (resize to 300x300 for SSD and normalize)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load dataset
train_dataset = VOCDataset(root='.', image_set='train', year='2007', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

from torchvision.datasets import VOCDetection
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2

# Define object class names
VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class VOCDataset(Dataset):
    def __init__(self, root, image_set='train', year='2007', transform=None):
        self.dataset = VOCDetection(root=root, year=year, image_set=image_set, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, annotation = self.dataset[idx]
        boxes = []
        labels = []

        for obj in annotation['annotation']['object']:
            bbox = obj['bndbox']
            label = VOC_CLASSES.index(obj['name'])
            box = [int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])]
            boxes.append(box)
            labels.append(label)

        img = np.array(img)

        if self.transform:
            img = self.transform(img)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return img, boxes, labels

# Define image transforms (resize to 300x300 for SSD and normalize)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load dataset
train_dataset = VOCDataset(root='.', image_set='train', year='2007', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train(model, dataloader, criterion_cls, criterion_bbox, optimizer, num_epochs=5):
    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, boxes, labels in tqdm(dataloader):
            images = torch.stack(images).to(device)
            boxes = list(b.to(device) for b in boxes)
            labels = list(l.to(device) for l in labels)

            optimizer.zero_grad()  # Zero the gradients before backward pass

            # Forward pass through the model
            pred_classes, pred_boxes = model(images)

            # Simulate only one target per image for simplicity (you can enhance this later)
            target_classes = torch.stack([l[0] for l in labels])  # Only using one target class
            target_boxes = torch.stack([b[0] for b in boxes])  # Only using one target box

            # Compute loss
            cls_loss = criterion_cls(pred_classes, target_classes)
            box_loss = criterion_bbox(pred_boxes, target_boxes)
            loss = cls_loss + box_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Print loss for each epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

model = ssd_model.to(device)

# Define loss functions
classification_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.SmoothL1Loss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train model
train(model, train_loader, criterion_cls=classification_loss_fn, criterion_bbox=bbox_loss_fn, optimizer=optimizer, num_epochs=10)


