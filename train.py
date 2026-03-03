import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import Sequence, to_categorical

# --- PHẦN 1: BỘ NÃO (Sửa num_metadata_features cho khớp) ---
def build_model(num_classes, num_metadata_features):
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False 
    
    # Nhánh ảnh
    img_input = layers.Input(shape=(224, 224, 3), name='input_image')
    x = base_model(img_input)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Nhánh metadata (Phải là 19 lỗ cắm)
    meta_input = layers.Input(shape=(num_metadata_features,), name='input_metadata')
    y = layers.Dense(32, activation='relu')(meta_input)
    y = layers.Dense(16, activation='relu')(y)

    # Kết hợp
    combined = layers.concatenate([x, y])
    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)
    output = layers.Dense(num_classes, activation='softmax')(z)
    
    return models.Model(inputs=[img_input, meta_input], outputs=output)

# --- PHẦN 2: BỘ NẠP (Sửa lại Tuple đầu ra) ---
class SkinDataGenerator(Sequence):
    def __init__(self, df, img_dir, batch_size=32, target_size=(224, 224), num_classes=7):
        self.df = df
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        self.meta_columns = [
            'age', 'sex_female', 'sex_male', 'sex_unknown', 
            'localization_abdomen', 'localization_acral', 'localization_back', 
            'localization_chest', 'localization_ear', 'localization_face', 
            'localization_foot', 'localization_genital', 'localization_hand', 
            'localization_lower extremity', 'localization_neck', 'localization_scalp', 
            'localization_trunk', 'localization_unknown', 'localization_upper extremity'
        ]

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_df = self.df[index * self.batch_size : (index + 1) * self.batch_size]
        images, metadata, labels = [], [], []

        for _, row in batch_df.iterrows():
            img_path = f"{self.img_dir}/{row['image_id']}.jpg"
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, self.target_size)
                img = img / 255.0
                meta_feat = row[self.meta_columns].values.astype('float32')
                
                images.append(img)
                metadata.append(meta_feat)
                labels.append(row['label'])

        # Quan trọng: Dùng ngoặc tròn () bao quanh [images, metadata]
        return (np.array(images, dtype='float32'), np.array(metadata, dtype='float32')), to_categorical(labels, num_classes=self.num_classes)

# --- PHẦN 3: CHẠY THỰC TẾ ---
# Kiểm tra lại tên file csv của ông nhé, tôi đang để theo tên phổ biến nhất
train_df = pd.read_csv('train_data.csv') 
test_df = pd.read_csv('test_data.csv')

# num_metadata_features PHẢI LÀ 19
model = build_model(num_classes=7, num_metadata_features=19)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Kiểm tra đường dẫn folder ảnh 'all_img'
train_gen = SkinDataGenerator(train_df, img_dir='all_img', batch_size=32)
val_gen = SkinDataGenerator(test_df, img_dir='all_img', batch_size=32)

print("Mọi thứ đã sẵn sàng. Bắt đầu nạp 19 đặc trưng và ảnh...")
model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1)

model.save('skin_model_v1.h5')
print("Xong! Đã lưu model.")