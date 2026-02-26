import pandas as pd
import os
from sklearn.model_selection import train_test_split
df = pd.read_csv('HAM10000_metadata.csv')
df['age']=df['age'].fillna(df['age'].mean())

path_img = './data/all_img/'

df["full_path"] = df["image_id"].apply(lambda x : os.path.join(path_img, x+".jpg"))

df_encoded = pd.get_dummies(df,columns=["sex","localization"],dtype=int)
df_encoded['label']=df_encoded["dx"].map({'nv':0, 'mel':1, 'bkl':2, 'bcc':3, 'akiec':4, 'vasc':5, 'df':6})
train_df,test_df = train_test_split(df_encoded, test_size=0.2, random_state=15, stratify=df_encoded['label'])
train_df.to_csv('train_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)