import tensorflow as tf
import numpy as np
import cv2

# 1. Load bộ não đã lưu
model = tf.keras.models.load_model('skin_model_pro.h5')

# 2. Chuẩn bị 1 tấm ảnh mới để "khám"
def predict_skin(img_path, metadata_list):
    # Xử lý ảnh y hệt lúc train
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0) # Biến thành (1, 224, 224, 3)

    # Xử lý metadata (19 con số)
    meta = np.array([metadata_list], dtype='float32')

    # 3. AI đưa ra dự đoán
    prediction = model.predict([img, meta])
    
    # 4. Giải mã kết quả
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'] # Tên 7 loại bệnh
    result_idx = np.argmax(prediction) # Lấy vị trí có xác suất cao nhất
    
    print(f"Dự đoán: {class_names[result_idx]}")
    print(f"Độ tin cậy: {prediction[0][result_idx] * 100:.2f}%")

# Chạy thử (Thay đường dẫn ảnh và 19 số metadata của ông vào đây)
predict_skin('./data/all_img/ISIC_0033979.jpg', [80.0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])