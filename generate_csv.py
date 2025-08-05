import os
import pandas as pd

# Folder where your dataset images are stored
DATASET_DIR = "dataset"

# Collect data
data = []

for filename in os.listdir(DATASET_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        try:
            name_parts = filename.split(".")[0].split("_")
            age = int(name_parts[0])
            gender = name_parts[1].lower()
            hair = name_parts[2].lower()

            image_path = os.path.join(DATASET_DIR, filename)

            data.append({
                "image": image_path,
                "age": age,
                "gender": gender,
                "hair_length": hair
            })

        except Exception as e:
            print(f"Skipped file: {filename} due to error: {e}")

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("processed.csv", index=False)

print(f"Processed {len(df)} records and saved to processed.csv")
