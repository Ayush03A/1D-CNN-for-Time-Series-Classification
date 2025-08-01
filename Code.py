import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import time

# 1. Generate synthetic dataset
def generate_signals(n_samples=1000, length=100):
    X = []
    y = []
    for _ in range(n_samples):
        t = np.linspace(0, 2 * np.pi, length)
        label = np.random.choice([0, 1])
        if label == 0:
            signal = np.sin(t) + np.random.normal(0, 0.2, length)  # Sine wave + noise
        else:
            signal = np.sign(np.sin(t)) + np.random.normal(0, 0.2, length)  # Square wave + noise
        X.append(signal)
        y.append(label)
    return np.array(X), np.array(y)

X, y = generate_signals(n_samples=2000, length=100)

# Shuffle the dataset
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

# Normalize
X = (X - X.mean()) / X.std()

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape for 1D CNN input: (samples, timesteps, channels)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# One-hot encode labels
y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)

# 2. Build 1D CNN model
model = models.Sequential([
    layers.Conv1D(32, 7, activation='relu', input_shape=(100, 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 5, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3. Callback to track training time
class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

# 4. Train the model
history = model.fit(
    X_train, y_train_cat,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[time_callback]
)

# 5. Plot training accuracy and time per epoch
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'b-o', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'r--s', label='Validation Accuracy')
plt.title('Model Accuracy Across Epochs', fontsize=14)
plt.xlabel('Epoch Number', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True)

# Time per epoch plot
plt.subplot(1, 2, 2)
plt.plot(range(1, len(time_callback.times) + 1), time_callback.times, 'g-D', linewidth=2)
plt.title('Training Time Per Epoch', fontsize=14)
plt.xlabel('Epoch Number', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.show()
