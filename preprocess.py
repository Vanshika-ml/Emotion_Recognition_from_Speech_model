import os
import pandas as pd

dataset_path = "dataset/dataset"

emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprised"
}

paths = []
emotions = []

for actor in os.listdir(dataset_path):

    actor_path = os.path.join(dataset_path, actor)

    if os.path.isdir(actor_path):

        for file in os.listdir(actor_path):

            if file.endswith(".wav"):

                emotion = file.split("-")[2]

                paths.append(os.path.join(actor_path, file))

                emotions.append(emotion_dict[emotion])

df = pd.DataFrame()

df["Path"] = paths
df["Emotion"] = emotions

print(df.head())

df.to_csv("emotion_data.csv", index=False)

print("CSV Saved Successfully")