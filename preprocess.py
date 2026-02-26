import pandas as pd
import os
df = pd.read_csv('HAM10000_metadata.csv')
df['age']=df['age'].fillna(df['age'].mean())

path_img = './data/all_img/'

df["full_path"] = df["image_id"].apply(lambda x : os.path.join(path_img, x+".jpg"))

df_encoded = pd.get_dummies(df,columns=["sex","localization"],dtype=int)
df_encoded['label']=df_encoded["dx"].map({'nv':0, 'mel':1, 'bkl':2, 'bcc':3, 'akiec':4, 'vasc':5, 'df':6})
print(df_encoded['label'])