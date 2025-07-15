# 07. Deep Learning

## 🎯 Learning Objectives
- Master neural network fundamentals and architectures
- Understand backpropagation and optimization techniques
- Learn specialized architectures (CNNs, RNNs, Transformers)
- Apply deep learning to real-world problems

---

## 1. Introduction to Deep Learning

**Deep Learning** is a subset of machine learning using artificial neural networks with multiple layers to progressively extract higher-level features from data.

### 1.1 What Makes It "Deep"? 🟢

#### Key Characteristics:
- **Multiple hidden layers**: Typically 3+ layers
- **Hierarchical feature learning**: Each layer learns increasingly complex features
- **End-to-end learning**: Learn features and classifier jointly
- **Non-linear transformations**: Can model complex relationships

#### Deep Learning vs Traditional ML:
```
Traditional ML: Raw Data → Feature Engineering → ML Algorithm → Prediction
Deep Learning: Raw Data → Neural Network → Prediction
```

### 1.2 Why Deep Learning Works 🟢

#### Universal Approximation Theorem:
Neural networks with sufficient width can approximate any continuous function.

#### Advantages of Depth:
- **Hierarchical representations**: Natural for many domains
- **Parameter efficiency**: Deeper networks need fewer parameters for same function
- **Compositional structure**: Matches real-world data structure

#### Recent Breakthroughs:
- **Big Data**: More training data available
- **Computational Power**: GPUs and specialized hardware
- **Algorithmic Improvements**: Better optimization and architectures
- **Transfer Learning**: Pre-trained models reduce training requirements

---

## 2. Neural Network Fundamentals

### 2.1 Perceptron 🟢

**The Building Block**: Single neuron with inputs, weights, and activation.

#### Mathematical Model:
```
y = f(Σᵢ wᵢxᵢ + b)
```

Where:
- xᵢ: Input features
- wᵢ: Weights
- b: Bias
- f: Activation function

#### Implementation:
```python
import numpy as np

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.01):
        self.weights = np.random.normal(0, 0.1, num_inputs)
        self.bias = 0
        self.learning_rate = learning_rate
    
    def forward(self, inputs):
        linear_output = np.dot(inputs, self.weights) + self.bias
        return self.activation(linear_output)
    
    def activation(self, x):
        return 1 if x > 0 else 0  # Step function
    
    def train(self, training_inputs, labels, epochs=100):
        for epoch in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.forward(inputs)
                error = label - prediction
                
                # Update weights and bias
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error
```

#### Limitations:
- Only linearly separable problems
- Cannot solve XOR problem
- Single decision boundary

### 2.2 Multilayer Perceptron (MLP) 🟢

**Solution**: Multiple layers of perceptrons with non-linear activations.

#### Architecture:
```
Input Layer → Hidden Layer(s) → Output Layer
```

#### Forward Propagation:
```python
import numpy as np

class MLP:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = {
                'weights': np.random.normal(0, 0.1, (layer_sizes[i], layer_sizes[i+1])),
                'bias': np.zeros((1, layer_sizes[i+1]))
            }
            self.layers.append(layer)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -709, 709)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        activations = [X]
        
        for layer in self.layers:
            z = np.dot(activations[-1], layer['weights']) + layer['bias']
            a = self.sigmoid(z)
            activations.append(a)
        
        return activations
```

### 2.3 Activation Functions 🟢

#### Common Activation Functions:

**Sigmoid:**
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Pros: Smooth, output in (0,1)
# Cons: Vanishing gradients, not zero-centered
```

**Tanh:**
```python
def tanh(x):
    return np.tanh(x)

# Pros: Zero-centered, smooth
# Cons: Vanishing gradients
```

**ReLU (Rectified Linear Unit):**
```python
def relu(x):
    return np.maximum(0, x)

# Pros: No vanishing gradients, computationally efficient
# Cons: Dead neurons, not differentiable at 0
```

**Leaky ReLU:**
```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Pros: Fixes dead ReLU problem
# Cons: Additional hyperparameter
```

**Swish/SiLU:**
```python
def swish(x):
    return x * sigmoid(x)

# Pros: Smooth, self-gated, often better than ReLU
```

#### Choosing Activation Functions:
- **Hidden layers**: ReLU (default), Leaky ReLU, Swish
- **Output layer**: 
  - Binary classification: Sigmoid
  - Multi-class classification: Softmax
  - Regression: Linear (no activation)

### 2.4 Backpropagation 🟡

**The Learning Algorithm**: Efficiently compute gradients using chain rule.

#### Chain Rule Application:
```
∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w
```

#### Backpropagation Implementation:
```python
def backward(self, X, y, learning_rate=0.01):
    m = X.shape[0]
    
    # Forward pass
    activations = self.forward(X)
    
    # Backward pass
    deltas = []
    
    # Output layer error
    output_error = activations[-1] - y
    deltas.append(output_error)
    
    # Hidden layer errors (working backwards)
    for i in range(len(self.layers) - 2, -1, -1):
        layer_error = np.dot(deltas[-1], self.layers[i+1]['weights'].T) * \
                     self.sigmoid_derivative(activations[i+1])
        deltas.append(layer_error)
    
    deltas.reverse()
    
    # Update weights and biases
    for i, layer in enumerate(self.layers):
        layer['weights'] -= learning_rate * np.dot(activations[i].T, deltas[i]) / m
        layer['bias'] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m

def train(self, X, y, epochs=1000, learning_rate=0.01):
    for epoch in range(epochs):
        self.backward(X, y, learning_rate)
        
        if epoch % 100 == 0:
            loss = self.compute_loss(X, y)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

def compute_loss(self, X, y):
    predictions = self.forward(X)[-1]
    return np.mean((predictions - y) ** 2)
```

#### Gradient Descent Variants:

**Batch Gradient Descent:**
```python
# Use entire dataset for each update
def batch_gradient_descent(self, X, y, learning_rate=0.01, epochs=1000):
    for epoch in range(epochs):
        self.backward(X, y, learning_rate)
```

**Stochastic Gradient Descent:**
```python
# Use one sample for each update
def sgd(self, X, y, learning_rate=0.01, epochs=1000):
    for epoch in range(epochs):
        for i in range(len(X)):
            self.backward(X[i:i+1], y[i:i+1], learning_rate)
```

**Mini-batch Gradient Descent:**
```python
# Use small batches for each update
def mini_batch_gd(self, X, y, batch_size=32, learning_rate=0.01, epochs=1000):
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            self.backward(batch_X, batch_y, learning_rate)
```

---

## 3. Modern Deep Learning Frameworks

### 3.1 PyTorch Implementation 🟢

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MLPTorch(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPTorch, self).__init__()
        
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # Add activation except for output layer
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Training loop
def train_pytorch_model(model, train_loader, val_loader, epochs=100):
    criterion = nn.MSELoss()  # or nn.CrossEntropyLoss() for classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}")

# Example usage
model = MLPTorch(input_size=784, hidden_sizes=[128, 64], output_size=10)
train_pytorch_model(model, train_loader, val_loader)
```

### 3.2 TensorFlow/Keras Implementation 🟢

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sequential API
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Functional API (for complex architectures)
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Compile and train
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
)
```

---

## 4. Convolutional Neural Networks (CNNs)

### 4.1 CNN Fundamentals 🟡

**Concept**: Networks designed for processing grid-like data (images, time series).

#### Key Components:

**Convolutional Layer:**
```
Output[i,j] = Σₘ Σₙ Input[i+m, j+n] × Kernel[m,n]
```

**Key Properties:**
- **Local connectivity**: Each neuron connects to local region
- **Parameter sharing**: Same weights used across spatial locations
- **Translation invariance**: Features detected regardless of position

#### Implementation:
```python
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
```

### 4.2 CNN Components 🟡

#### Convolution Operation:
```python
def convolution_2d(image, kernel, stride=1, padding=0):
    """Simple 2D convolution implementation"""
    # Add padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant')
    
    h, w = image.shape
    kh, kw = kernel.shape
    
    # Output dimensions
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(0, out_h * stride, stride):
        for j in range(0, out_w * stride, stride):
            output[i//stride, j//stride] = np.sum(
                image[i:i+kh, j:j+kw] * kernel
            )
    
    return output
```

#### Pooling Operations:
```python
def max_pooling(feature_map, pool_size=2, stride=2):
    """Max pooling operation"""
    h, w = feature_map.shape
    
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            start_i = i * stride
            start_j = j * stride
            pool_region = feature_map[start_i:start_i+pool_size, 
                                     start_j:start_j+pool_size]
            output[i, j] = np.max(pool_region)
    
    return output

def average_pooling(feature_map, pool_size=2, stride=2):
    """Average pooling operation"""
    # Similar to max pooling but use np.mean instead of np.max
    pass
```

### 4.3 Popular CNN Architectures 🔴

#### LeNet-5 (1998):
```python
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

#### AlexNet (2012):
```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
```

#### ResNet (Residual Networks):
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = F.relu(out)
        return out
```

---

## 5. Recurrent Neural Networks (RNNs)

### 5.1 RNN Fundamentals 🟡

**Concept**: Networks with memory that can process sequences of variable length.

#### Basic RNN:
```
h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)
y_t = W_hy × h_t + b_y
```

#### Implementation:
```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_hy = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size)
        
        outputs = []
        for t in range(seq_len):
            hidden = torch.tanh(self.W_xh(x[:, t]) + self.W_hh(hidden))
            output = self.W_hy(hidden)
            outputs.append(output)
        
        return torch.stack(outputs, dim=1), hidden
```

#### Vanilla RNN Problems:
- **Vanishing gradients**: Gradients become very small
- **Exploding gradients**: Gradients become very large
- **Short-term memory**: Difficulty learning long-term dependencies

### 5.2 Long Short-Term Memory (LSTM) 🟡

**Solution**: Gating mechanisms to control information flow.

#### LSTM Gates:
```python
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Forget gate
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Input gate
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_C = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Output gate
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x, hidden):
        h_prev, C_prev = hidden
        
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)
        
        # Forget gate
        f_t = torch.sigmoid(self.W_f(combined))
        
        # Input gate
        i_t = torch.sigmoid(self.W_i(combined))
        C_tilde = torch.tanh(self.W_C(combined))
        
        # Update cell state
        C_t = f_t * C_prev + i_t * C_tilde
        
        # Output gate
        o_t = torch.sigmoid(self.W_o(combined))
        h_t = o_t * torch.tanh(C_t)
        
        return h_t, (h_t, C_t)
```

#### PyTorch LSTM:
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use last time step output
        output = self.fc(lstm_out[:, -1, :])
        return output
```

### 5.3 Gated Recurrent Unit (GRU) 🟡

**Simpler Alternative**: Fewer parameters than LSTM, often similar performance.

```python
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Reset gate
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Update gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        
        # New gate
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x, hidden):
        combined = torch.cat([x, hidden], dim=1)
        
        # Reset gate
        r_t = torch.sigmoid(self.W_r(combined))
        
        # Update gate
        z_t = torch.sigmoid(self.W_z(combined))
        
        # New gate
        combined_new = torch.cat([x, r_t * hidden], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_new))
        
        # Update hidden state
        h_t = (1 - z_t) * hidden + z_t * h_tilde
        
        return h_t
```

### 5.4 Sequence-to-Sequence Models 🔴

**Applications**: Machine translation, text summarization, chatbots.

#### Encoder-Decoder Architecture:
```python
class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size):
        super(Seq2Seq, self).__init__()
        
        # Encoder
        self.encoder_embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Decoder
        self.decoder_embedding = nn.Embedding(output_vocab_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_projection = nn.Linear(hidden_size, output_vocab_size)
    
    def encode(self, src):
        embedded = self.encoder_embedding(src)
        outputs, (hidden, cell) = self.encoder_lstm(embedded)
        return hidden, cell
    
    def decode(self, tgt, hidden, cell):
        embedded = self.decoder_embedding(tgt)
        outputs, (hidden, cell) = self.decoder_lstm(embedded, (hidden, cell))
        predictions = self.output_projection(outputs)
        return predictions, hidden, cell
    
    def forward(self, src, tgt):
        hidden, cell = self.encode(src)
        predictions, _, _ = self.decode(tgt, hidden, cell)
        return predictions
```

---

## 6. Attention Mechanisms and Transformers

### 6.1 Attention Mechanism 🔴

**Problem**: Fixed-size context vector in seq2seq models creates bottleneck.

#### Basic Attention:
```python
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        
        seq_len = encoder_outputs.size(1)
        
        # Repeat hidden state for each encoder output
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Calculate attention energies
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attention_weights = F.softmax(self.v(energy).squeeze(2), dim=1)
        
        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        
        return context.squeeze(1), attention_weights
```

### 6.2 Self-Attention 🔴

**Key Innovation**: Attention within the same sequence.

#### Scaled Dot-Product Attention:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size must be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Calculate attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out
```

### 6.3 Transformer Architecture 🔴

**Revolutionary**: Attention is all you need - no recurrence or convolution.

#### Transformer Block:
```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        # Add & Norm
        x = self.dropout(self.norm1(attention + query))
        
        forward = self.feed_forward(x)
        
        # Add & Norm
        out = self.dropout(self.norm2(forward + x))
        return out
```

#### Full Transformer:
```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,
                 embed_size=256, num_layers=6, forward_expansion=4, heads=8,
                 dropout=0.1, max_length=100):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, 
                              heads, forward_expansion, dropout, max_length)
        
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers,
                              heads, forward_expansion, dropout, max_length)
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
```

---

## 7. Optimization and Regularization

### 7.1 Advanced Optimizers 🟡

#### Adam Optimizer:
```python
class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0
    
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
```

#### Learning Rate Scheduling:
```python
# PyTorch schedulers
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Custom scheduler
def cosine_annealing(epoch, max_epochs, lr_max, lr_min=0):
    return lr_min + (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / max_epochs)) / 2
```

### 7.2 Regularization Techniques 🟡

#### Dropout:
```python
class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
    
    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
            return x * mask / (1 - self.p)
        return x
```

#### Batch Normalization:
```python
class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * batch_var
            
            # Normalize
            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # Use running statistics during inference
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        return self.weight * x_norm + self.bias
```

#### Weight Decay (L2 Regularization):
```python
# In PyTorch optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Manual implementation
def l2_regularization(model, lambda_reg):
    l2_loss = 0
    for param in model.parameters():
        l2_loss += torch.norm(param, 2) ** 2
    return lambda_reg * l2_loss
```

### 7.3 Advanced Training Techniques 🔴

#### Gradient Clipping:
```python
def clip_gradients(model, max_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# Usage in training loop
loss.backward()
clip_gradients(model, max_norm=1.0)
optimizer.step()
```

#### Mixed Precision Training:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 🎯 Key Takeaways

### Architecture Selection Guide:

#### For Image Data:
- **Simple classification**: CNN (ResNet, EfficientNet)
- **Object detection**: YOLO, R-CNN family
- **Segmentation**: U-Net, DeepLab
- **Generation**: GAN, VAE

#### For Sequential Data:
- **Short sequences**: LSTM, GRU
- **Long sequences**: Transformer
- **Time series**: LSTM, GRU, Temporal CNNs
- **Language**: Transformer (BERT, GPT)

#### For Tabular Data:
- **Simple**: MLP with dropout
- **Complex**: TabNet, Neural ODEs
- **Mixed types**: Wide & Deep networks

### Best Practices:
1. **Start simple**: Begin with basic architectures
2. **Data first**: More/better data often beats complex models
3. **Regularization**: Use dropout, batch norm, weight decay
4. **Monitoring**: Track training/validation metrics
5. **Hyperparameter tuning**: Learning rate is most important
6. **Transfer learning**: Use pre-trained models when possible

---

## 📚 Next Steps

Continue your deep learning journey with:
- **[Natural Language Processing](08_Natural_Language_Processing.md)** - Text and language models
- **[Computer Vision](09_Computer_Vision.md)** - Advanced image processing techniques

---

## 🛠️ Practical Exercises

### Exercise 1: Build CNN from Scratch
1. Implement convolution and pooling layers manually
2. Build a simple CNN for MNIST
3. Compare with PyTorch implementation
4. Analyze feature maps and filters

### Exercise 2: LSTM for Time Series
1. Create synthetic time series data
2. Implement LSTM for forecasting
3. Compare with simple RNN and GRU
4. Add attention mechanism

### Exercise 3: Transformer Implementation
1. Implement multi-head attention
2. Build encoder-decoder transformer
3. Train on sequence-to-sequence task
4. Visualize attention weights

---

*Next: [Natural Language Processing →](08_Natural_Language_Processing.md)*
