import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tensorflow.keras.models import load_model
from feature_extraction import extract_features

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Speech Emotion Recognition using Deep Learning")
st.markdown("Upload a **WAV** audio file and let the AI detect the speaker's emotion.")

st.sidebar.title("📌 Project Information")
st.sidebar.write("**Dataset:** RAVDESS")
st.sidebar.write("**Feature Extraction:** MFCC")
st.sidebar.write("**Model:** CNN")
st.sidebar.write("**Framework:** TensorFlow/Keras")

emotion_labels = [
    "angry", "calm", "disgust", "fear",
    "happy", "neutral", "sad", "surprised"
]

emoji = {
    "angry": "😠", "calm": "😌", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprised": "😲"
}

@st.cache_resource
def load_emotion_model():
    return load_model("emotion_model.h5")

model = load_emotion_model()

uploaded_file = st.file_uploader("📂 Upload WAV File", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    temp_path = "temp.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    audio, sr = librosa.load(temp_path, sr=None)

    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr)
    ax.set_title("Audio Waveform")
    st.pyplot(fig)

    try:
        feature = extract_features(temp_path)
        feature = np.expand_dims(feature, axis=0)   # shape: (1, 130, 40)


        prediction = model.predict(feature, verbose=0)
        confidence = prediction[0] * 100

        top3 = np.argsort(confidence)[::-1][:3]
        st.subheader("🏆 Top 3 Predictions")
        for i in top3:
            st.write(f"**{emotion_labels[i].capitalize()}** : {confidence[i]:.2f}%")

        predicted_index = np.argmax(prediction)
        emotion = emotion_labels[predicted_index]
        st.success(f"## 🎯 Predicted Emotion: {emotion.capitalize()} {emoji[emotion]}")

        confidence_df = pd.DataFrame({
            "Emotion": [e.capitalize() for e in emotion_labels],
            "Confidence(%)": prediction[0] * 100
        })

        st.subheader("📊 Emotion Prediction Confidence")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(emotion_labels, confidence)
        ax.set_ylabel("Confidence(%)")
        ax.set_title("Emotion Prediction Confidence")
        plt.xticks(rotation=30)
        st.pyplot(fig)

        st.info("✅ Prediction completed successfully!")

    except Exception as e:
        st.error(f"⚠️ Could not process this audio file: {e}")