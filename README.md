# 🌊 1D-CNN for Time Series Classification

This repository showcases the power of **1D Convolutional Neural Networks (1D-CNNs)** for classifying time series data. The model is built using TensorFlow/Keras to distinguish between synthetic noisy sine waves and noisy square waves, a common task in signal processing.

---

## 🌟 Objective

The goal is to demonstrate the effectiveness and efficiency of 1D-CNNs for sequence data analysis. This project aims to:

- 🏗️ Build a 1D-CNN model from scratch using Keras.
- 📉 Differentiate between two distinct types of noisy waveforms (sine vs. square).
- ⚡ Highlight the speed advantage of CNNs over traditional recurrent models for certain sequential tasks.
- 📊 Visualize model accuracy and training time to evaluate performance.

---

## 🚀 Key Concepts & Features

- **1D-CNN Architecture**
  A Keras `Sequential` model designed specifically for 1D data.

- **Parallel Processing Power**
  Unlike RNNs, CNNs process data in parallel, making them incredibly fast on GPUs.

- **Fundamental Building Blocks**
  -  convolutions `Conv1D`: Acts as a powerful feature extractor for local temporal patterns (like rising/falling edges).
  - pooling `MaxPooling1D`: Downsamples the sequence, creating robustness to small shifts.
  - 🌍 `GlobalMaxPooling1D`: Aggregates features across the entire sequence for a final classification vector.
  - activation `ReLU` & `Softmax`: For non-linearity and final probability output.

- **Synthetic Data Generation**
  - 📈 The script generates its own training data, making it completely self-contained.

---

## 📊 Dataset: Synthetic Waveforms 📈

- **Type**: Synthetic 1D time series data generated via Python.
- **Classes**: 2
  - **Sine Waves**: `signal = sin(t) + N(0, 0.2)`
  - **Square Waves**: `signal = sign(sin(t)) + N(0, 0.2)`
- **Samples**: 2,000 (1,000 per class)
- **Sequence Length**: 100 time steps per sample.

---

## 🛠️ How to Run

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Ayush03A/1D-CNN-for-Time-Series-Classification.git
    cd [1D-CNN-for-Time-Series-Classification]
    ```

2.  **Install Dependencies**
    ```bash
    pip install tensorflow numpy matplotlib
    ```

3.  **Run the Script**
    This will generate the data, train the 1D-CNN, and display the performance plots.

---

## ✅ Conclusion

1D-CNNs are a highly effective and computationally efficient tool for time series classification. This project proves that they can achieve **near-perfect accuracy** with incredible speed, learning to distinguish complex temporal patterns. This makes them an excellent choice for real-world applications like **ECG analysis**, **audio classification**, and **industrial sensor monitoring**.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
