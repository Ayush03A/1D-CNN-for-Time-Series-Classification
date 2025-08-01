import numpy as np‬
‭ import tensorflow as tf‬
‭ from tensorflow.keras import layers, models‬
‭ from tensorflow.keras.utils import to
_
categorical‬
‭ import matplotlib.pyplot as plt‬
‭ import time‬
‭ # 1. Generate synthetic dataset‬
‭ def generate
_
signals(n
_
‭ X = []‬
‭ y = []‬
‭ for
samples=1000, length=100):‬
_
in range(n
_
samples):‬
‭ t = np.linspace(0, 2*np.pi, length)‬
‭ # Randomly choose class‬
‭ label = np.random.choice([0, 1])‬
‭ if label == 0:‬
‭ else:‬
‭ X.append(signal)‬
‭ y.append(label)‬
‭ return np.array(X), np.array(y)‬
‭ signal = np.sin(t) + np.random.normal(0, 0.2, length) # Sine wave + noise‬
‭ signal = np.sign(np.sin(t)) + np.random.normal(0, 0.2, length) # Square wave + noise‬
‭ X, y = generate
_
signals(n
_
samples=2000, length=100)‬
‭ # Shuffle‬
‭ idx = np.random.permutation(len(X))‬
‭ X, y = X[idx], y[idx]‬
‭ # Normalize‬
‭ X = (X - X.mean()) / X.std()‬
‭ # Train-test split‬
‭ split = int(0.8 * len(X))‬
‭ X
train, X
_
_
test = X[:split], X[split:]‬
‭ y_
train, y_
test = y[:split], y[split:]‬
‭ # Reshape for 1D CNN: (samples, timesteps, channels)‬
‭ X
‭ X
train = X
_
_
train[..., np.newaxis]‬
test = X
_
_
test[..., np.newaxis]‬
‭ # One-hot encode labels‬
‭ y_
train
cat = to
_
_
categorical(y_
train, 2)‬
‭ y_
test
cat = to
_
_
categorical(y_
test, 2)‬
‭ # 2. Build 1D CNN model‬
‭ model = models.Sequential([‬
‭ layers.Conv1D(32, 7, activation='relu'
, input
_
‭ layers.MaxPooling1D(2),‬
‭ layers.Conv1D(64, 5, activation='relu'),‬
‭ layers.GlobalMaxPooling1D(),‬
‭ layers.Dense(64, activation='relu'),‬
‭ layers.Dropout(0.5),‬
‭ layers.Dense(2, activation='softmax')‬
shape=(100, 1)),‬
‭ ])‬
‭ model.compile(optimizer='adam'
, loss='categorical
_
crossentropy'
, metrics=['accuracy'])‬
‭ # 3. Callback for timing‬
‭ class TimeHistory(tf.keras.callbacks.Callback):‬
‭ def
init
__
__(self):‬
‭ super().
init
__
__()‬
‭ self.times = []‬
‭ def on
_
epoch
_
begin(self, epoch, logs=None):‬
‭ self.epoch
time
_
_
start = time.time()‬
‭ def on
_
epoch
_
end(self, epoch, logs=None):‬
‭ self.times.append(time.time() - self.epoch
time
_
_
‭ time
_
callback = TimeHistory()‬
‭ # 4. Train the model‬
‭ history = model.fit(‬
‭ X
_
train, y_
train
cat,‬
_
‭ epochs=10,‬
‭ batch
size=64,‬
_
‭ validation
_
split=0.2,‬
‭ callbacks=[time
_
callback]‬
start)‬
‭ )‬
‭ # 5. Plot training accuracy and training time per epoch‬
‭ plt.figure(figsize=(12,5))‬
‭ plt.subplot(1,2,1)‬
‭ plt.plot(history.history['accuracy'],
'b-o'
, label='Training Accuracy')‬
‭ plt.plot(history.history['val
_
accuracy'],
'r--s'
, label='Validation Accuracy')‬
‭ plt.title('Model Accuracy Across Epochs'
, fontsize=14)‬
‭ plt.xlabel('Epoch Number'
, fontsize=12)‬
‭ plt.ylabel('Accuracy'
, fontsize=12)‬
‭ plt.legend()‬
‭ plt.grid(True)‬
‭ plt.subplot(1,2,2)‬
‭ plt.plot(range(1, len(time
_
callback.times)+1), time
_
‭ plt.title('Training Time Per Epoch'
, fontsize=14)‬
‭ plt.xlabel('Epoch Number'
, fontsize=12)‬
‭ plt.ylabel('Time (seconds)'
, fontsize=12)‬
‭ plt.grid(True)‬
callback.times,
'g-D'
, linewidth=2)‬
‭ plt.tight
_
‭ plt.show()‬
layout(
