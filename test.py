import pandas as pd
from feature_extraction import extract_features

df = pd.read_csv("emotion_data.csv")

feature = extract_features(df["Path"][0])

print("Feature Shape:", feature.shape)
print(feature)