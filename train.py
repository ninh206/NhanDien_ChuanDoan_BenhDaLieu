import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import Sequence, to_categorical

# --- PHẦN 1: BỘ NẠO DỮ LIỆU (ĐÃ THÊM LẠI VÀO ĐÂY) ---
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

        return (np.array(images, dtype='float32'), np.array(metadata, dtype='float32')), to_categorical(labels, num_classes=self.num_classes)

# --- PHẦN 2: ĐỊNH NGHĨA BỘ NÃO ULTIMATE ---
def build_model(num_classes, num_metadata_features):
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False 
    
    img_input = layers.Input(shape=(224, 224, 3), name='input_image')
    
    # Tăng cường dữ liệu để AI bớt "ngáo" khi gặp ảnh nhiễu
    aug = layers.RandomFlip("horizontal_and_vertical")(img_input)
    aug = layers.RandomRotation(0.2)(aug)
    aug = layers.RandomContrast(0.1)(aug)
    
    x = base_model(aug)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x) 
    x = layers.Dropout(0.4)(x) 
    
    meta_input = layers.Input(shape=(num_metadata_features,), name='input_metadata')
    y = layers.Dense(64, activation='relu')(meta_input)
    y = layers.Dropout(0.2)(y)

    combined = layers.concatenate([x, y])
    z = layers.Dense(128, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)
    output = layers.Dense(num_classes, activation='softmax')(z)
    
    return models.Model(inputs=[img_input, meta_input], outputs=output)

# --- PHẦN 3: CHIẾN DỊCH HUẤN LUYỆN ---
# Kiểm tra file csv của Ninh xem tên đúng là train_df hay train_data nhé!
train_df = pd.read_csv('train_df.csv') 
test_df = pd.read_csv('test_df.csv')

model = build_model(num_classes=7, num_metadata_features=19)
train_gen = SkinDataGenerator(train_df, img_dir='./data/all_img', batch_size=32)
val_gen = SkinDataGenerator(test_df, img_dir='./data/all_img', batch_size=32)

# GIAI ĐOẠN 1: HUẤN LUYỆN LỚP NGOÀI
print("\n>>> BẮT ĐẦU GIAI ĐOẠN 1: LÀM NÓNG")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=10)

# GIAI ĐOẠN 2: FINE-TUNING ĐẠI PHẪU
print("\n>>> BẮT ĐẦU GIAI ĐOẠN 2: TINH CHỈNH SÂU")
# Tìm layer MobileNetV2 trong model (ở đây là layer index 4)
for layer in model.layers:
    if "mobilenetv2" in layer.name:
        base_model = layer
        break

base_model.trainable = True
# Mở khóa từ lớp 50 trở đi để AI soi kỹ hơn
for layer in base_model.layers[:50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)
model.fit(train_gen, validation_data=val_gen, epochs=15)

model.save('skin_model_ultimate.h5')
print("\nXong! Bộ não Ultimate đã sẵn sàng. Chúc mừng Ninh!")