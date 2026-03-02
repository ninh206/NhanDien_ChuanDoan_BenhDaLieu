import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_model(num_classes,num_metadata_features):
    #NHánh 1: Xử lý ảnh (DÙng MobileNetV2)
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Đóng băng các lớp của MobileNetV2
    img_input = layers.Input(shape=(224, 224, 3), name='input_image')
    x = base_model(img_input)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    #Nhánh 2 : Xử lý triệu chứng (Dùng Dense làm tai)
    meta_input = layers.Input(shape=(num_metadata_features,), name='input_metadata')
    y = layers.Dense(32, activation='relu')(meta_input)
    y = layers.Dense(16, activation='relu')(y)

    #Kết hợp 2 nhánh
    combined = layers.concatenate([x, y])
    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)

    output = layers.Dense(num_classes, activation='softmax')(z)
    model = models.Model(inputs=[img_input, meta_input], outputs=output)
    return model

model = build_model(num_classes=7, num_metadata_features=10)
model.summary()