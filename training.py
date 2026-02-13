import pandas as pd
import os
df = pd.read_csv('HAM10000_metadata.csv')
img_folder = 'data/all_img/' 
df["full_path"] = df["image_id"].apply(lambda x: os.path.join(img_folder, x + ".jpg"))
# 1. Điền giá trị tuổi còn thiếu (NaN) bằng tuổi trung bình
df['age'] = df['age'].fillna(df['age'].mean())

# 2. Mã hóa các triệu chứng văn bản (sex, localization) thành số
# Kỹ thuật One-Hot Encoding: Biến 'male/female' thành các cột 0 và 1
df_encoded = pd.get_dummies(df, columns=['sex', 'localization'])

# 3. Chuyển nhãn bệnh (dx) thành số để máy học 
# Ví dụ: nv -> 0, mel -> 1, bkl -> 2...
df_encoded['label'] = df_encoded['dx'].astype('category').cat.codes
from sklearn.model_selection import train_test_split

# Chia df_encoded thành 2 bảng: 80% để train, 20% để test
# stratify=df_encoded['label'] giúp giữ nguyên tỷ lệ các loại bệnh
train_df, val_df = train_test_split(
    df_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=df_encoded['label']
)

features_columns = [col for col in train_df.columns if col not in ['image_id', 'dx', 'label', 'full_path']]
import tensorflow as tf
# Xóa dòng 29 cũ, dùng trực tiếp từ tf
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

# Định nghĩa cách xử lý ảnh (Scale giá trị điểm ảnh về khoảng 0-1)
datagen = ImageDataGenerator(rescale=1./255)

def my_generator(df, batch_size=32):
    # Nhánh nạp ảnh từ đường dẫn 'full_path'
    gen = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='full_path',
        y_col='label',
        target_size=(224, 224), # Kích thước ảnh chuẩn cho CNN
        batch_size=batch_size,
        class_mode='raw',
        shuffle=True
    )
    
    while True:
        data = next(gen)
        # Lấy chỉ số (index) của các dòng ảnh vừa được nạp
        idx = gen.index_array
        # Lấy dữ liệu triệu chứng (Metadata) tương ứng
        metadata = df.iloc[idx][features_columns].values
        
        # Trả về đồng thời: ([Mảng ảnh, Mảng triệu chứng], Nhãn bệnh)
        yield [data[0], metadata], data[1]

# Tạo ra 2 bộ nạp cho tập học và tập thi
train_gen = my_generator(train_df)
val_gen = my_generator(val_df)