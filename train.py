import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from feature_extraction import extract_features, MAX_LEN, N_MFCC
from model import build_model

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

df = pd.read_csv("emotion_data.csv")

X, y = [], []
for index, row in df.iterrows():
    try:
        X.append(extract_features(row["Path"]))
        y.append(row["Emotion"])
    except Exception as e:
        print(f"⚠️ Skipping {row['Path']}: {e}")

X = np.array(X)   # shape: (num_samples, MAX_LEN, N_MFCC)
y = np.array(y)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=SEED,
    stratify=y_encoded
)

# No reshape needed now -- already (samples, time_steps, n_mfcc)

model = build_model(input_shape=(MAX_LEN, N_MFCC), num_classes=len(encoder.classes_))
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

model.save("emotion_model.h5")
print("Model Saved Successfully")

plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.savefig("accuracy.png")
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()
plt.savefig("loss.png")
plt.close()

predictions = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

print("\nClassification Report:\n")
print(classification_report(y_test, predictions, target_names=encoder.classes_))