# 🎤 Speech Emotion Recognition using Deep Learning

A Deep Learning-based Speech Emotion Recognition system that predicts human emotions from speech audio using a Convolutional Neural Network (CNN). The application is built with TensorFlow/Keras and deployed using Streamlit.

---

## 🚀 Features

- 🎵 Upload WAV audio files
- 🎯 Predicts 8 different emotions
- 🏆 Displays Top-3 predicted emotions
- 📊 Emotion confidence bar chart
- 🌊 Audio waveform visualization
- 🤖 CNN-based Deep Learning model
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
- Scikit-learn

---

## 📂 Dataset

**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**

Dataset contains professionally recorded emotional speech audio for training and evaluation.

---

## 📁 Project Structure

```
Emotion_Recognition_from_Speech_model/
│
├── dataset/
├── images/
│   ├── home.png
        waveform.png
        prediction_result.png
│   ├── prediction_confidence.png
│   ├── accuracy.png
│   ├── loss.png
│   └── confusion_matrix.png
│
├── feature_extraction.py
├── preprocess.py
├── train.py
├── model.py
├── test.py
├── app.py
├── emotion_data.csv
├── requirements.txt
├── README.md

```

---

## 📊 Model Performance

- Model : CNN
- Number of Classes : 8
- Dataset : RAVDESS
- Feature Extraction : MFCC
- Optimizer : Adam
- Loss Function : Sparse Categorical Crossentropy

---

# 📷 Screenshots

## 🏠 Home Page

![Home](images/home.png)

---

## Audio Waveform

![Waveform](images/waveform.png)

---

## Emotion Prediction Result

![Result](images/prediction_result.png)

---

## 📊 Emotion Prediction Confidence

![Confidence](images/prediction_confidence.png)

---

## 📈 Training Accuracy

![Accuracy](images/accuracy.png)

---

## 📉 Training Loss

![Loss](images/loss.png)

---

## 📊 Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Emotion_Recognition_from_Speech_model.git
```

Move into project directory

```bash
cd Emotion_Recognition_from_Speech_model
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit App

```bash
streamlit run app.py
```

---

## 📌 Future Improvements

- Mel Spectrogram Visualization
- LSTM + CNN Hybrid Model
- Real-time Microphone Prediction
- Model Comparison
- Deployment on Streamlit Cloud
- Better UI/UX

---

## 👩‍💻 Author

**Vanshika Varshney**

GitHub:
https://github.com/Vanshika-ml

---

⭐ If you found this project useful, don't forget to star the repository.
