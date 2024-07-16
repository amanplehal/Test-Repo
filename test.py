import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
import cv2

# Define transformations for the training data
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load your dataset (example uses VOC)
dataset = VOCDetection(root='data/VOC', year='2012', image_set='train', download=True, transform=transform)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# Define the model (pre-trained Faster R-CNN for simplicity)
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (gesture) + background

# Modify the classifier head
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)

# Move model to GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in dataloader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f'Epoch: {epoch}, Loss: {losses.item()}')

# Save the model
torch.save(model.state_dict(), 'gesture_detection_model.pth')

