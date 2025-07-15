# Deep Learning and Neural Networks

## Table of Contents
1. [Neural Network Fundamentals](#neural-network-fundamentals)
2. [TensorFlow/Keras Advanced](#tensorflowkeras-advanced)
3. [PyTorch Advanced](#pytorch-advanced)
4. [Computer Vision](#computer-vision)
5. [Natural Language Processing](#natural-language-processing)
6. [Model Deployment](#model-deployment)

## Neural Network Fundamentals

### 1. Perceptron and Multi-layer Perceptron
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split

# Simple Perceptron Implementation
class Perceptron:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
    def fit(self, X, y):
        # Initialize weights and bias
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)
        
        # Training
        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)
                
                # Update weights and bias
                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
    
    def activation_function(self, x):
        return np.where(x >= 0, 1, -1)
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = self.activation_function(linear_output)
        return predictions

# Test Perceptron
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

perceptron = Perceptron()
perceptron.fit(X_train, y_train)
predictions = perceptron.predict(X_test)

accuracy = np.mean(predictions == np.where(y_test <= 0, -1, 1))
print(f"Perceptron Accuracy: {accuracy:.3f}")
```

### 2. Activation Functions
```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # Clip to prevent overflow

def sigmoid_derivative(x):
    """Derivative of sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """Hyperbolic tangent activation function"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh"""
    return 1 - np.tanh(x)**2

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function"""
    return np.where(x > 0, x, alpha * x)

def swish(x):
    """Swish activation function"""
    return x * sigmoid(x)

def gelu(x):
    """GELU activation function"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# Visualize activation functions
x = np.linspace(-5, 5, 100)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(x, sigmoid(x), label='Sigmoid', color='blue')
plt.plot(x, sigmoid_derivative(x), label='Sigmoid Derivative', color='red', linestyle='--')
plt.title('Sigmoid')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(x, tanh(x), label='Tanh', color='blue')
plt.plot(x, tanh_derivative(x), label='Tanh Derivative', color='red', linestyle='--')
plt.title('Tanh')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(x, relu(x), label='ReLU', color='blue')
plt.plot(x, relu_derivative(x), label='ReLU Derivative', color='red', linestyle='--')
plt.title('ReLU')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(x, leaky_relu(x), label='Leaky ReLU', color='blue')
plt.title('Leaky ReLU')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(x, swish(x), label='Swish', color='blue')
plt.title('Swish')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(x, gelu(x), label='GELU', color='blue')
plt.title('GELU')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 3. Neural Network from Scratch
```python
class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        """
        Initialize neural network
        layers: list of integers representing number of neurons in each layer
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            # Xavier initialization
            weight = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / layers[i])
            bias = np.zeros((1, layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)
    
    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Apply activation function (ReLU for hidden layers, Sigmoid for output)
            if i == len(self.weights) - 1:
                activation = sigmoid(z)
            else:
                activation = relu(z)
            
            self.activations.append(activation)
        
        return self.activations[-1]
    
    def backward(self, X, y, output):
        """Backward propagation"""
        m = X.shape[0]
        
        # Calculate output layer error
        dz = output - y
        dw = (1/m) * np.dot(self.activations[-2].T, dz)
        db = (1/m) * np.sum(dz, axis=0, keepdims=True)
        
        # Store gradients
        dw_list = [dw]
        db_list = [db]
        
        # Propagate error backwards
        for i in range(len(self.weights) - 2, -1, -1):
            dz = np.dot(dz, self.weights[i + 1].T) * relu_derivative(self.z_values[i])
            dw = (1/m) * np.dot(self.activations[i].T, dz)
            db = (1/m) * np.sum(dz, axis=0, keepdims=True)
            
            dw_list.insert(0, dw)
            db_list.insert(0, db)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dw_list[i]
            self.biases[i] -= self.learning_rate * db_list[i]
    
    def train(self, X, y, epochs):
        """Train the neural network"""
        losses = []
        
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)
            
            # Calculate loss (binary cross-entropy)
            loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
            losses.append(loss)
            
            # Backward propagation
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return (output > 0.5).astype(int)

# Test the neural network
# Generate non-linearly separable data
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.3, random_state=42)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train network
nn = NeuralNetwork([2, 10, 5, 1], learning_rate=0.1)
losses = nn.train(X_train, y_train, epochs=1000)

# Make predictions
predictions = nn.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy:.3f}")
```

## TensorFlow/Keras Advanced

### 1. Custom Layers and Models
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Custom Layer
class CustomDenseLayer(layers.Layer):
    def __init__(self, units, activation=None):
        super(CustomDenseLayer, self).__init__()
        self.units = units
        self.activation = activation
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        output = tf.matmul(inputs, self.w) + self.b
        if self.activation:
            output = self.activation(output)
        return output

# Custom Model
class CustomModel(keras.Model):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.dense1 = CustomDenseLayer(128, activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(0.3)
        self.dense2 = CustomDenseLayer(64, activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(0.3)
        self.dense3 = CustomDenseLayer(num_classes, activation=tf.nn.softmax)
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.dense3(x)

# Test custom model
model = CustomModel(10)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Generate sample data
X_sample = tf.random.normal((100, 20))
y_sample = tf.random.uniform((100,), maxval=10, dtype=tf.int32)

model.fit(X_sample, y_sample, epochs=5, verbose=1)
```

### 2. Transfer Learning
```python
# Transfer Learning with pre-trained models
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load pre-trained VGG16 model
base_model = VGG16(
    weights='imagenet',  # Pre-trained on ImageNet
    include_top=False,   # Exclude the final classification layer
    input_shape=(224, 224, 3)
)

# Freeze base model layers
base_model.trainable = False

# Add custom classification head
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')  # 10 classes
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Transfer Learning Model:")
model.summary()

# Fine-tuning: Unfreeze some layers
def fine_tune_model(model, base_model, unfreeze_layers=10):
    """Fine-tune by unfreezing top layers"""
    base_model.trainable = True
    
    # Freeze all layers except the last few
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    
    # Use a lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Example of fine-tuning (uncomment to use)
# fine_tuned_model = fine_tune_model(model, base_model)
```

### 3. Advanced Training Techniques
```python
# Custom Training Loop with tf.GradientTape
import tensorflow as tf

class AdvancedTrainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        # Metrics
        self.train_loss = keras.metrics.Mean()
        self.train_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.val_loss = keras.metrics.Mean()
        self.val_accuracy = keras.metrics.SparseCategoricalAccuracy()
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.loss_fn(y, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss(loss)
        self.train_accuracy(y, predictions)
        
        return loss
    
    @tf.function
    def val_step(self, x, y):
        predictions = self.model(x, training=False)
        loss = self.loss_fn(y, predictions)
        
        self.val_loss(loss)
        self.val_accuracy(y, predictions)
        
        return loss
    
    def train(self, train_dataset, val_dataset, epochs):
        for epoch in range(epochs):
            # Reset metrics
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()
            
            # Training
            for x_batch, y_batch in train_dataset:
                self.train_step(x_batch, y_batch)
            
            # Validation
            for x_batch, y_batch in val_dataset:
                self.val_step(x_batch, y_batch)
            
            print(f"Epoch {epoch + 1}")
            print(f"Train Loss: {self.train_loss.result():.4f}, "
                  f"Train Acc: {self.train_accuracy.result():.4f}")
            print(f"Val Loss: {self.val_loss.result():.4f}, "
                  f"Val Acc: {self.val_accuracy.result():.4f}")

# Example usage with MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Create model
simple_model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10)
])

# Create trainer
trainer = AdvancedTrainer(
    model=simple_model,
    optimizer=keras.optimizers.Adam(),
    loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

# Train model (uncomment to run)
# trainer.train(train_dataset, val_dataset, epochs=5)
```

## PyTorch Advanced

### 1. Custom Datasets and DataLoaders
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return torch.FloatTensor(sample), torch.LongTensor([label])

# Create sample dataset
np.random.seed(42)
data = np.random.randn(1000, 10)
labels = np.random.randint(0, 3, 1000)

# Split data
train_size = int(0.8 * len(data))
train_data, test_data = data[:train_size], data[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]

# Create datasets
train_dataset = CustomDataset(train_data, train_labels)
test_dataset = CustomDataset(test_data, test_labels)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")
```

### 2. Custom Neural Network Module
```python
class CustomNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super(CustomNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Create model
model = CustomNeuralNetwork(
    input_size=10,
    hidden_sizes=[128, 64, 32],
    num_classes=3,
    dropout_rate=0.3
)

print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            targets = targets.squeeze()  # Remove extra dimension
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

# Test function
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            targets = targets.squeeze()
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=20)

# Test the model
test_accuracy = test_model(model, test_loader)
```

### 3. Advanced PyTorch Features
```python
# Learning Rate Scheduling
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

# Different schedulers
step_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=50)

# Model checkpointing
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

# Example usage
# save_checkpoint(model, optimizer, epoch, loss, 'model_checkpoint.pth')
# model, optimizer, start_epoch, loss = load_checkpoint('model_checkpoint.pth', model, optimizer)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

# Mixed Precision Training
from torch.cuda.amp import GradScaler, autocast

def train_with_mixed_precision(model, train_loader, criterion, optimizer, epochs):
    scaler = GradScaler()
    
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            targets = targets.squeeze()
            
            optimizer.zero_grad()
            
            # Use autocast for forward pass
            with autocast():
                outputs = model(data)
                loss = criterion(outputs, targets)
            
            # Scale loss and backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

# Data Parallel Training (multiple GPUs)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model.to(device)
```

## Computer Vision

### 1. Convolutional Neural Networks (CNN)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AdvancedCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten and fully connected
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x

# Data augmentation and preprocessing
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Create model
cnn_model = AdvancedCNN(num_classes=10)
print(f"Model parameters: {sum(p.numel() for p in cnn_model.parameters()):,}")
```

### 2. Object Detection and Segmentation
```python
# Object Detection with YOLO-style architecture (simplified)
class YOLOBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YOLOBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, 1)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        return x

class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=20, num_anchors=3):
        super(SimpleYOLO, self).__init__()
        
        # Backbone (simplified)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            YOLOBlock(32, 64),
            nn.MaxPool2d(2, 2),
            
            YOLOBlock(64, 128),
            nn.MaxPool2d(2, 2),
            
            YOLOBlock(128, 256),
            nn.MaxPool2d(2, 2),
            
            YOLOBlock(256, 512),
        )
        
        # Detection head
        self.detection_head = nn.Conv2d(512, num_anchors * (5 + num_classes), 1)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.detection_head(x)
        return x

# Semantic Segmentation with U-Net architecture
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = UNetBlock(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = UNetBlock(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        
        # Output
        self.out_conv = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return torch.sigmoid(self.out_conv(dec1))

# Create models
yolo_model = SimpleYOLO(num_classes=20)
unet_model = UNet(in_channels=3, num_classes=1)

print(f"YOLO parameters: {sum(p.numel() for p in yolo_model.parameters()):,}")
print(f"U-Net parameters: {sum(p.numel() for p in unet_model.parameters()):,}")
```

## 7. Advanced Neural Network Architectures

### Vision Transformers (ViT)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, 
                                   kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.projection(x)
        # Flatten patches: (B, embed_dim, n_patches)
        x = x.flatten(2)
        # Transpose: (B, n_patches, embed_dim)
        x = x.transpose(1, 2)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, embed_dim=768, n_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.projection(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer implementation"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 num_classes=1000, embed_dim=768, depth=12, n_heads=12, 
                 mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Learnable parameters
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Classification head (use only class token)
        x = self.head(x[:, 0])
        return x

# Create ViT model
vit = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=10,
    embed_dim=768,
    depth=12,
    n_heads=12
)

print(f"ViT parameters: {sum(p.numel() for p in vit.parameters()):,}")

# Test forward pass
dummy_input = torch.randn(2, 3, 224, 224)
output = vit(dummy_input)
print(f"Output shape: {output.shape}")
```

### Generative Adversarial Networks (GANs)

```python
class Generator(nn.Module):
    """DCGAN Generator"""
    
    def __init__(self, latent_dim=100, img_channels=3, feature_maps=64):
        super().__init__()
        
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            
            # State: (feature_maps * 8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            # State: (feature_maps * 4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # State: (feature_maps * 2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            # State: feature_maps x 32 x 32
            nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: img_channels x 64 x 64
        )
    
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    """DCGAN Discriminator"""
    
    def __init__(self, img_channels=3, feature_maps=64):
        super().__init__()
        
        self.main = nn.Sequential(
            # Input: img_channels x 64 x 64
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: feature_maps x 32 x 32
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (feature_maps * 2) x 16 x 16
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (feature_maps * 4) x 8 x 8
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (feature_maps * 8) x 4 x 4
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

# Training setup for GAN
def weights_init(m):
    """Initialize weights for GAN"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Create GAN
latent_dim = 100
generator = Generator(latent_dim)
discriminator = Discriminator()

# Apply weight initialization
generator.apply(weights_init)
discriminator.apply(weights_init)

# Loss function and optimizers
criterion = nn.BCELoss()
lr = 0.0002
beta1 = 0.5

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

# Training function
def train_gan_step(real_data, generator, discriminator, optimizer_G, optimizer_D, criterion, device):
    """Single training step for GAN"""
    batch_size = real_data.size(0)
    
    # Labels
    real_label = 1.0
    fake_label = 0.0
    
    # Train Discriminator
    optimizer_D.zero_grad()
    
    # Real data
    output_real = discriminator(real_data)
    label_real = torch.full((batch_size,), real_label, device=device)
    loss_D_real = criterion(output_real, label_real)
    loss_D_real.backward()
    
    # Fake data
    noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    fake_data = generator(noise)
    output_fake = discriminator(fake_data.detach())
    label_fake = torch.full((batch_size,), fake_label, device=device)
    loss_D_fake = criterion(output_fake, label_fake)
    loss_D_fake.backward()
    
    optimizer_D.step()
    
    # Train Generator
    optimizer_G.zero_grad()
    
    output_fake = discriminator(fake_data)
    loss_G = criterion(output_fake, label_real)  # Generator wants to fool discriminator
    loss_G.backward()
    
    optimizer_G.step()
    
    return loss_D_real.item() + loss_D_fake.item(), loss_G.item()
```

## 8. Advanced Training Techniques

### Mixed Precision Training

```python
import torch
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    """Trainer with automatic mixed precision"""
    
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = GradScaler()
    
    def train_step(self, data, target):
        self.optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            output = self.model(data)
            loss = self.criterion(output, target)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = vit.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

trainer = MixedPrecisionTrainer(model, optimizer, criterion, device)
```

### Gradient Accumulation and Clipping

```python
class AdvancedTrainer:
    """Advanced trainer with gradient accumulation and clipping"""
    
    def __init__(self, model, optimizer, criterion, device, 
                 accumulation_steps=4, max_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.scaler = GradScaler()
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        
        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target) / self.accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(dataloader):
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
        
        return total_loss / len(dataloader)

# Learning rate scheduling
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine learning rate schedule with warmup"""
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Example with advanced training
import math

advanced_trainer = AdvancedTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    accumulation_steps=4,
    max_grad_norm=1.0
)

# Learning rate scheduler
num_epochs = 10
num_training_steps = num_epochs * 100  # Assuming 100 steps per epoch
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps
)
```

## 9. Model Optimization and Deployment

### Model Quantization

```python
import torch.quantization as quantization

def quantize_model(model, dataloader):
    """Quantize model for inference optimization"""
    
    # Prepare model for quantization
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    model_prepared = quantization.prepare(model, inplace=False)
    
    # Calibration
    model_prepared.eval()
    with torch.no_grad():
        for data, _ in dataloader:
            model_prepared(data)
    
    # Convert to quantized model
    model_quantized = quantization.convert(model_prepared, inplace=False)
    
    return model_quantized

# Dynamic quantization (simpler approach)
def dynamic_quantize_model(model):
    """Apply dynamic quantization"""
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

# Model pruning
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    """Prune model weights"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
    
    return model

# Knowledge distillation
class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation loss function"""
    
    def __init__(self, temperature=3.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_outputs, teacher_outputs, targets):
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(student_outputs, targets)
        
        # Knowledge distillation loss
        student_soft = F.log_softmax(student_outputs / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)
        kd_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        return total_loss

# Example usage
def train_student_with_teacher(student_model, teacher_model, dataloader, num_epochs=5):
    """Train student model with knowledge distillation"""
    
    kd_loss = KnowledgeDistillationLoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
    
    teacher_model.eval()
    
    for epoch in range(num_epochs):
        student_model.train()
        total_loss = 0.0
        
        for data, targets in dataloader:
            optimizer.zero_grad()
            
            # Get predictions
            student_outputs = student_model(data)
            
            with torch.no_grad():
                teacher_outputs = teacher_model(data)
            
            # Calculate loss
            loss = kd_loss(student_outputs, teacher_outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
```

This completes the comprehensive deep learning and neural networks documentation with state-of-the-art architectures, advanced training techniques, and deployment optimization strategies.
