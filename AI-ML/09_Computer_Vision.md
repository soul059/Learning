# 09. Computer Vision

## 🎯 Learning Objectives
- Understand image processing fundamentals and techniques
- Master convolutional neural networks for image analysis
- Learn object detection and recognition algorithms
- Apply computer vision to real-world problems

---

## 1. Introduction to Computer Vision

**Computer Vision** is a field of AI that trains computers to interpret and understand visual information from the world.

### 1.1 What is Computer Vision? 🟢

#### Core Tasks:
- **Image Classification**: What is in the image?
- **Object Detection**: Where are objects in the image?
- **Semantic Segmentation**: Which pixels belong to which objects?
- **Instance Segmentation**: Separate individual instances of objects
- **Image Generation**: Create new images
- **Face Recognition**: Identify specific individuals
- **Optical Character Recognition (OCR)**: Extract text from images

#### Applications:
- **Autonomous Vehicles**: Self-driving cars
- **Medical Imaging**: X-ray, MRI analysis
- **Security**: Surveillance, access control
- **Retail**: Visual search, inventory management
- **Manufacturing**: Quality control, defect detection
- **Agriculture**: Crop monitoring, disease detection

### 1.2 Image Fundamentals 🟢

#### Digital Images:
```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Load and display image
def load_and_display_image(image_path):
    # Using PIL
    img_pil = Image.open(image_path)
    
    # Using OpenCV
    img_cv2 = cv2.imread(image_path)
    img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    
    # Using matplotlib
    img_plt = plt.imread(image_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_pil)
    axes[0].set_title('PIL Image')
    axes[0].axis('off')
    
    axes[1].imshow(img_cv2_rgb)
    axes[1].set_title('OpenCV Image')
    axes[1].axis('off')
    
    axes[2].imshow(img_plt)
    axes[2].set_title('Matplotlib Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return img_cv2_rgb

# Image properties
def analyze_image_properties(image):
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image size: {image.size}")
    print(f"Min value: {image.min()}")
    print(f"Max value: {image.max()}")
    print(f"Mean value: {image.mean():.2f}")

# Create sample image
def create_sample_image():
    # Create a simple gradient image
    height, width = 100, 100
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            image[i, j] = [i * 255 // height, j * 255 // width, (i + j) * 255 // (height + width)]
    
    return image

sample_img = create_sample_image()
analyze_image_properties(sample_img)
```

#### Color Spaces:
```python
def demonstrate_color_spaces(image):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original RGB
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('RGB')
    axes[0, 0].axis('off')
    
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    # HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    axes[0, 2].imshow(hsv)
    axes[0, 2].set_title('HSV')
    axes[0, 2].axis('off')
    
    # Individual RGB channels
    axes[1, 0].imshow(image[:, :, 0], cmap='Reds')
    axes[1, 0].set_title('Red Channel')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(image[:, :, 1], cmap='Greens')
    axes[1, 1].set_title('Green Channel')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(image[:, :, 2], cmap='Blues')
    axes[1, 2].set_title('Blue Channel')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Usage
# demonstrate_color_spaces(sample_img)
```

---

## 2. Image Processing Techniques

### 2.1 Basic Image Operations 🟢

#### Geometric Transformations:
```python
def geometric_transformations(image):
    height, width = image.shape[:2]
    
    # Rotation
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    # Scaling
    scaled = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    
    # Translation
    translation_matrix = np.float32([[1, 0, 50], [0, 1, 30]])
    translated = cv2.warpAffine(image, translation_matrix, (width, height))
    
    # Flipping
    horizontal_flip = cv2.flip(image, 1)
    vertical_flip = cv2.flip(image, 0)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    transformations = [
        (image, 'Original'),
        (rotated, 'Rotated 45°'),
        (scaled, 'Scaled 0.5x'),
        (translated, 'Translated'),
        (horizontal_flip, 'Horizontal Flip'),
        (vertical_flip, 'Vertical Flip')
    ]
    
    for i, (img, title) in enumerate(transformations):
        row, col = i // 3, i % 3
        axes[row, col].imshow(img if len(img.shape) == 3 else img, 
                             cmap='gray' if len(img.shape) == 2 else None)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Perspective transformation
def perspective_transform(image):
    height, width = image.shape[:2]
    
    # Define source points (corners of the image)
    src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    
    # Define destination points (trapezoid shape)
    dst_points = np.float32([[50, 0], [width-50, 0], [0, height], [width, height]])
    
    # Calculate perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply transformation
    transformed = cv2.warpPerspective(image, matrix, (width, height))
    
    return transformed
```

#### Filtering and Enhancement:
```python
def apply_filters(image):
    # Convert to grayscale for some operations
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Gaussian blur
    gaussian_blur = cv2.GaussianBlur(image, (15, 15), 0)
    
    # Median filter (good for salt-and-pepper noise)
    median_filtered = cv2.medianBlur(image, 5)
    
    # Bilateral filter (edge-preserving)
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Edge detection
    edges_canny = cv2.Canny(gray, 100, 200)
    
    # Sobel edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Sharpening
    sharpening_kernel = np.array([[-1, -1, -1],
                                 [-1, 9, -1],
                                 [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, sharpening_kernel)
    
    # Display results
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    filters = [
        (image, 'Original'),
        (gaussian_blur, 'Gaussian Blur'),
        (median_filtered, 'Median Filter'),
        (bilateral, 'Bilateral Filter'),
        (edges_canny, 'Canny Edges'),
        (sobel_combined, 'Sobel Edges'),
        (sharpened, 'Sharpened'),
        (gray, 'Grayscale'),
        (image, 'Original')
    ]
    
    for i, (img, title) in enumerate(filters):
        row, col = i // 3, i % 3
        if len(img.shape) == 2:
            axes[row, col].imshow(img, cmap='gray')
        else:
            axes[row, col].imshow(img)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
```

### 2.2 Feature Detection 🟡

#### Corner Detection:
```python
def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Harris corner detection
    harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    # Dilate to mark corners
    harris_corners = cv2.dilate(harris_corners, None)
    
    # Mark corners on original image
    image_harris = image.copy()
    image_harris[harris_corners > 0.01 * harris_corners.max()] = [255, 0, 0]
    
    # Shi-Tomasi corner detection
    corners_shi_tomasi = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    
    image_shi_tomasi = image.copy()
    if corners_shi_tomasi is not None:
        corners_shi_tomasi = np.int0(corners_shi_tomasi)
        for corner in corners_shi_tomasi:
            x, y = corner.ravel()
            cv2.circle(image_shi_tomasi, (x, y), 3, (0, 255, 0), -1)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(image_harris)
    axes[1].set_title('Harris Corners')
    axes[1].axis('off')
    
    axes[2].imshow(image_shi_tomasi)
    axes[2].set_title('Shi-Tomasi Corners')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# SIFT (Scale-Invariant Feature Transform)
def detect_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # Draw keypoints
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, 
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    print(f"Number of SIFT keypoints detected: {len(keypoints)}")
    print(f"Descriptor shape: {descriptors.shape if descriptors is not None else 'None'}")
    
    return image_with_keypoints, keypoints, descriptors

# ORB (Oriented FAST and Rotated BRIEF)
def detect_orb_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create ORB detector
    orb = cv2.ORB_create()
    
    # Detect keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    # Draw keypoints
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    
    print(f"Number of ORB keypoints detected: {len(keypoints)}")
    
    return image_with_keypoints, keypoints, descriptors
```

---

## 3. Convolutional Neural Networks (CNNs)

### 3.1 CNN Fundamentals 🟡

#### Basic CNN Architecture:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Assuming 32x32 input
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# More advanced CNN with batch normalization
class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AdvancedCNN, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Second block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Third block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# Training function
def train_cnn(model, train_loader, val_loader, epochs=10, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        train_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
        
        scheduler.step()

# Data preparation for CIFAR-10
def prepare_cifar10_data(batch_size=32):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

# Usage
# train_loader, test_loader = prepare_cifar10_data()
# model = AdvancedCNN(num_classes=10)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train_cnn(model, train_loader, test_loader, epochs=10, device=device)
```

### 3.2 Transfer Learning 🟡

#### Using Pre-trained Models:
```python
import torchvision.models as models
from torchvision import transforms

class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, model_name='resnet18', pretrained=True):
        super(TransferLearningModel, self).__init__()
        
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        
        elif model_name == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
            num_features = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Linear(num_features, num_classes)
        
        elif model_name == 'efficientnet':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
        
        elif model_name == 'mobilenet':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# Feature extraction (freeze backbone)
def setup_feature_extraction(model, feature_extracting=True):
    if feature_extracting:
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        # Only train the classifier
        if hasattr(model.backbone, 'fc'):
            for param in model.backbone.fc.parameters():
                param.requires_grad = True
        elif hasattr(model.backbone, 'classifier'):
            for param in model.backbone.classifier.parameters():
                param.requires_grad = True

# Fine-tuning with different learning rates
def setup_fine_tuning_optimizer(model, backbone_lr=1e-4, classifier_lr=1e-3):
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': classifier_params, 'lr': classifier_lr}
    ])
    
    return optimizer

# Example usage
def create_transfer_learning_model(num_classes, model_name='resnet18', 
                                 feature_extraction=True):
    model = TransferLearningModel(num_classes, model_name, pretrained=True)
    
    if feature_extraction:
        setup_feature_extraction(model, feature_extracting=True)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                          model.parameters()), lr=0.001)
    else:
        optimizer = setup_fine_tuning_optimizer(model)
    
    return model, optimizer
```

---

## 4. Object Detection

### 4.1 Traditional Object Detection 🟡

#### Template Matching:
```python
def template_matching(image, template):
    # Convert images to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    
    # Template matching methods
    methods = [
        cv2.TM_CCOEFF_NORMED,
        cv2.TM_CCORR_NORMED,
        cv2.TM_SQDIFF_NORMED
    ]
    
    method_names = [
        'Correlation Coefficient',
        'Cross Correlation',
        'Squared Difference'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, (method, name) in enumerate(zip(methods, method_names)):
        # Apply template matching
        result = cv2.matchTemplate(image_gray, template_gray, method)
        
        # Find best match location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # For SQDIFF methods, minimum is the best match
        if method == cv2.TM_SQDIFF_NORMED:
            top_left = min_loc
        else:
            top_left = max_loc
        
        # Calculate bottom right corner
        h, w = template_gray.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        # Draw rectangle on image
        result_image = image.copy()
        cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), 2)
        
        # Display results
        axes[0, i].imshow(result, cmap='gray')
        axes[0, i].set_title(f'{name} - Match Result')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(result_image)
        axes[1, i].set_title(f'{name} - Detection')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Cascade classifiers (Haar cascades)
def detect_faces_haar(image):
    # Load pre-trained face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around faces
    result_image = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Detect eyes within face region
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(result_image, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
    
    print(f"Detected {len(faces)} faces")
    return result_image
```

### 4.2 Modern Object Detection 🔴

#### YOLO (You Only Look Once) Implementation:
```python
import torch
import torch.nn as nn

class YOLOv1(nn.Module):
    def __init__(self, num_classes=20, num_boxes=2):
        super(YOLOv1, self).__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        
        # Backbone (simplified)
        self.backbone = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3
            nn.Conv2d(192, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Additional layers...
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 7 * 7 * (num_boxes * 5 + num_classes))
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.detection_head(x)
        
        # Reshape to grid format
        batch_size = x.size(0)
        x = x.view(batch_size, 7, 7, self.num_boxes * 5 + self.num_classes)
        
        return x

# YOLO Loss Function
class YOLOLoss(nn.Module):
    def __init__(self, num_classes=20, num_boxes=2):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
    
    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        
        # Split predictions
        # predictions: [batch, 7, 7, num_boxes*5 + num_classes]
        
        # Extract components
        pred_boxes = predictions[:, :, :, :self.num_boxes*5].contiguous()
        pred_classes = predictions[:, :, :, self.num_boxes*5:].contiguous()
        
        # Calculate individual losses
        coord_loss = self.calculate_coord_loss(pred_boxes, targets)
        conf_loss = self.calculate_conf_loss(pred_boxes, targets)
        class_loss = self.calculate_class_loss(pred_classes, targets)
        
        total_loss = (self.lambda_coord * coord_loss + 
                     conf_loss + 
                     self.lambda_noobj * class_loss)
        
        return total_loss
    
    def calculate_coord_loss(self, pred_boxes, targets):
        # Simplified coordinate loss calculation
        # This should be implemented based on the YOLO paper
        return torch.tensor(0.0)
    
    def calculate_conf_loss(self, pred_boxes, targets):
        # Simplified confidence loss calculation
        return torch.tensor(0.0)
    
    def calculate_class_loss(self, pred_classes, targets):
        # Simplified classification loss calculation
        return torch.tensor(0.0)

# Non-Maximum Suppression
def non_max_suppression(boxes, scores, threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping boxes
    
    Args:
        boxes: Tensor of shape [N, 4] containing bounding boxes
        scores: Tensor of shape [N] containing confidence scores
        threshold: IoU threshold for suppression
    
    Returns:
        keep: Indices of boxes to keep
    """
    if boxes.size(0) == 0:
        return torch.empty((0,), dtype=torch.long)
    
    # Sort boxes by scores
    _, indices = scores.sort(descending=True)
    
    keep = []
    while indices.size(0) > 0:
        # Keep the box with highest score
        current = indices[0].item()
        keep.append(current)
        
        if indices.size(0) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current].unsqueeze(0)
        remaining_boxes = boxes[indices[1:]]
        
        iou = calculate_iou(current_box, remaining_boxes)
        
        # Keep boxes with IoU below threshold
        indices = indices[1:][iou.squeeze() < threshold]
    
    return torch.tensor(keep, dtype=torch.long)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    # box format: [x1, y1, x2, y2]
    
    # Calculate intersection
    x1 = torch.max(box1[:, 0:1], box2[:, 0:1])
    y1 = torch.max(box1[:, 1:2], box2[:, 1:2])
    x2 = torch.min(box1[:, 2:3], box2[:, 2:3])
    y2 = torch.min(box1[:, 3:4], box2[:, 3:4])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate areas
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection
    
    return intersection / union
```

#### Using Pre-trained Object Detection Models:
```python
import torchvision.transforms as T
from torchvision.models import detection

def load_pretrained_detector(model_name='fasterrcnn_resnet50_fpn'):
    """Load pre-trained object detection model"""
    
    if model_name == 'fasterrcnn_resnet50_fpn':
        model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif model_name == 'maskrcnn_resnet50_fpn':
        model = detection.maskrcnn_resnet50_fpn(pretrained=True)
    elif model_name == 'retinanet_resnet50_fpn':
        model = detection.retinanet_resnet50_fpn(pretrained=True)
    
    model.eval()
    return model

def detect_objects(model, image, threshold=0.5):
    """Detect objects in image using pre-trained model"""
    
    # Preprocessing
    transform = T.Compose([T.ToTensor()])
    
    # Convert PIL image to tensor
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    input_tensor = transform(image).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        predictions = model(input_tensor)
    
    # Filter predictions by threshold
    pred = predictions[0]
    keep = pred['scores'] > threshold
    
    boxes = pred['boxes'][keep].numpy()
    labels = pred['labels'][keep].numpy()
    scores = pred['scores'][keep].numpy()
    
    return boxes, labels, scores

def visualize_detections(image, boxes, labels, scores, class_names=None):
    """Visualize object detection results"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        
        # Draw bounding box
        rect = plt.Rectangle((x1, y1), width, height,
                           linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        label_text = f'{class_names[label] if class_names else label}: {score:.2f}'
        ax.text(x1, y1 - 5, label_text, fontsize=10, color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    ax.axis('off')
    plt.title('Object Detection Results')
    plt.tight_layout()
    plt.show()

# COCO class names
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Example usage
# model = load_pretrained_detector('fasterrcnn_resnet50_fpn')
# boxes, labels, scores = detect_objects(model, image, threshold=0.5)
# visualize_detections(image, boxes, labels, scores, COCO_CLASSES)
```

---

## 5. Image Segmentation

### 5.1 Semantic Segmentation 🔴

#### U-Net Architecture:
```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (downsampling)
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder (upsampling)
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature))
        
        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse the list
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Transpose convolution
            skip_connection = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)  # Double convolution
        
        return self.final_conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

# Segmentation loss functions
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou

# Training function for segmentation
def train_segmentation_model(model, train_loader, val_loader, epochs=100, device='cpu'):
    criterion = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    model.to(device)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Combined loss
            bce_loss = criterion(outputs, masks)
            dice = dice_loss(outputs, masks)
            loss = bce_loss + dice
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                bce_loss = criterion(outputs, masks)
                dice = dice_loss(outputs, masks)
                loss = bce_loss + dice
                
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        scheduler.step(avg_val_loss)
```

---

## 🎯 Key Takeaways

### Computer Vision Pipeline:
1. **Data Preparation**: Proper image preprocessing and augmentation
2. **Model Selection**: Choose appropriate architecture for the task
3. **Training Strategy**: Transfer learning, fine-tuning, or training from scratch
4. **Evaluation**: Use appropriate metrics (accuracy, IoU, mAP)
5. **Deployment**: Optimize for inference speed and memory usage

### Best Practices:
- **Data Augmentation**: Essential for generalizable models
- **Transfer Learning**: Start with pre-trained models when possible
- **Proper Evaluation**: Use validation sets and cross-validation
- **Computational Efficiency**: Consider model size and inference speed
- **Domain Adaptation**: Fine-tune for specific applications

---

## 📚 Next Steps

Continue your journey with:
- **[Time Series Analysis](10_Time_Series_Analysis.md)** - Learn temporal data analysis
- **[Ensemble Methods](11_Ensemble_Methods.md)** - Combine multiple models

---

*Next: [Time Series Analysis →](10_Time_Series_Analysis.md)*
