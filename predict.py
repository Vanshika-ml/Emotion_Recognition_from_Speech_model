import argparse
import numpy as np
from tensorflow.keras.models import load_model
from feature_extraction import extract_features

emotion_labels = [
    "angry", "calm", "disgust", "fear",
    "happy", "neutral", "sad", "surprised"
]

def predict_emotion(audio_path, model_path="emotion_model.h5"):
    model = load_model(model_path)
    feature = extract_features(audio_path)          # shape: (130, 40)
    feature = np.expand_dims(feature, axis=0)        # shape: (1, 130, 40)
    prediction = model.predict(feature, verbose=0)
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion, prediction[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion from a WAV file")
    parser.add_argument("audio_path", help="Path to the WAV audio file")
    parser.add_argument("--model", default="emotion_model.h5", help="Path to trained model")
    args = parser.parse_args()

    emotion, confidence = predict_emotion(args.audio_path, args.model)
    print(f"Predicted Emotion: {emotion}")
    for label, conf in zip(emotion_labels, confidence * 100):
        print(f"  {label}: {conf:.2f}%")