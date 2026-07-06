import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from feature_extraction import extract_features
from model import build_model
# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load CSV
df = pd.read_csv("emotion_data.csv")

X = []
y = []

# Feature Extraction
for index, row in df.iterrows():
   try:  
      X.append(extract_features(row["Path"]))
      y.append(row["Emotion"])
   except Exception as e:
        print(f"⚠️ Skipping {row['Path']}: {e}")
    
X = np.array(X)
y = np.array(y)

# Encode Labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=SEED,
    stratify=y_encoded
)

# Reshape for CNN
X_train = X_train.reshape(X_train.shape[0], 40, 1)
X_test = X_test.reshape(X_test.shape[0], 40, 1)

# Build Model
model = build_model(num_classes=len(encoder.class))

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

print(f"Test Accuracy: {accuracy: .4f}")

# Save Model
model.save("emotion_model.h5")

print("Model Saved Successfully")

# Accuracy Graph
plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.savefig("accuracy.png")
plt.close()

# Loss Graph
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
plt.close()

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    predictions,
    target_names=encoder.classes_
))
