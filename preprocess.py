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
skipped = 0
for actor in os.listdir(dataset_path):

    actor_path = os.path.join(dataset_path, actor)

    if not os.path.isdir(actor_path):
        continue
    for file in os.listdir(actor_path):

        if not file.endswith(".wav"):
            continue
        try:    
                emotion_code = file.split("-")[2]
                emotion = emotion_dict[emotion_code]
                paths.append(os.path.join(actor_path, file))

                emotions.append(emotion)
        except (IndexError, KeyError):
            skipped += 1
            print(f"⚠️ Skipping malformed filename: {file}")


df = pd.DataFrame()

df["Path"] = paths
df["Emotion"] = emotions

print(df.head())
print(f"\nTotal files processed: {len(df)} | Skipped: {skipped}")

df.to_csv("emotion_data.csv", index=False)

print("CSV Saved Successfully")
