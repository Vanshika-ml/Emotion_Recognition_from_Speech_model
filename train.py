import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from feature_extraction import extract_features
from model import build_model

# Load CSV
df = pd.read_csv("emotion_data.csv")

X = []
y = []

# Feature Extraction
for index, row in df.iterrows():
    X.append(extract_features(row["Path"]))
    y.append(row["Emotion"])

X = np.array(X)
y = np.array(y)

# Encode Labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Reshape for CNN
X_train = X_train.reshape(X_train.shape[0], 40, 1)
X_test = X_test.reshape(X_test.shape[0], 40, 1)

# Build Model
model = build_model()

# Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)

# Save Model
model.save("emotion_model.h5")

print("Model Saved Successfully")

# Accuracy Graph
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.savefig("accuracy.png")
plt.show()

# Loss Graph
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()
plt.savefig("loss.png")
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report


predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)


cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8,6))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    predictions,
    target_names=encoder.classes_
))