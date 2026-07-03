import numpy as np
from tensorflow.keras.models import load_model
from feature_extraction import extract_features

emotion_labels = [
    "angry",
    "calm",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprised"
]

model = load_model("emotion_model.h5")

audio_path = input("Enter audio file path: ")

feature = extract_features(audio_path)
feature = feature.reshape(1, 40, 1)

prediction = model.predict(feature)

emotion = emotion_labels[np.argmax(prediction)]

print("Predicted Emotion:", emotion)