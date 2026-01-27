import pandas as pd
import os
df = pd.read_csv('HAM10000_metadata.csv')
img_folder = 'data/all_img/' 
df["full_path"] = df["image_id"].apply(lambda x: os.path.join(img_folder, x + ".jpg"))
print(df['full_path'].head())