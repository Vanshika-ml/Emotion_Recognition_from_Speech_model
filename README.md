# 🎤 Speech Emotion Recognition using Deep Learning

A Deep Learning-based Speech Emotion Recognition system that predicts human emotions from speech audio using a CNN + LSTM hybrid architecture. The model captures both local spectral patterns (via CNN) and temporal dependencies (via LSTM) in MFCC feature sequences. The application is built with TensorFlow/Keras and deployed using Streamlit.

---

## 🚀 Features

- 🎵 Upload WAV audio files
- 🎯 Predicts 8 different emotions
- 🏆 Displays Top-3 predicted emotions
- 📊 Emotion confidence bar chart
- 🌊 Audio waveform visualization
- 🤖 CNN + LSTM hybrid Deep Learning model
- 💻 Interactive Streamlit Web App
- 📈 Training Accuracy & Loss visualization
- 📉 Confusion Matrix
- ⚡ Real-time emotion prediction

---

## 🧠 Predicted Emotions

- 😠 Angry
- 😌 Calm
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprised

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- Librosa
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📂 Dataset

**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**

- 1440 audio samples
- 24 professional actors (12 male, 12 female)
- 8 emotion classes per actor

---

## 🧩 Model Architecture

Instead of averaging MFCC features into a single vector, this project extracts a **time-series MFCC sequence** (130 time-steps × 40 coefficients) per audio clip, preserving temporal information.

- **Conv1D layers** — learn local spectral patterns within MFCC frames
- **LSTM layer** — learn temporal dependencies across time-steps (how emotion evolves over the utterance)
- **Dense + Dropout layers** — final classification with regularization

---

## 📁 Project Structure
