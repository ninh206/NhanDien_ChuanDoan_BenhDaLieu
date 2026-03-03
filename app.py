import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd

# 1. Cấu hình trang web
st.set_page_config(page_title="AI Chẩn Đoán Da Liễu", layout="centered")
st.title("🩺 Bác sĩ AI Chẩn Đoán Bệnh Da Liễu")

# 2. Nạp bộ não AI
@st.cache_resource # Để không phải nạp lại model mỗi lần bấm nút
def load_my_model():
    return tf.keras.models.load_model('skin_model_v1.h5') # Hoặc skin_model_pro.h5

model = load_my_model()

# 3. Giao diện nhập thông tin (Bên trái màn hình)
st.sidebar.header("Thông tin bệnh nhân")
age = st.sidebar.slider("Tuổi", 0, 100, 25)
gender = st.sidebar.selectbox("Giới tính", ["Nữ", "Nam", "Không rõ"])
loc = st.sidebar.selectbox("Vị trí vết thương", 
    ['abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot', 
     'genital', 'hand', 'lower extremity', 'neck', 'scalp', 'trunk', 'unknown', 'upper extremity'])

# 4. Phần Upload ảnh (Chính diện)
uploaded_file = st.file_uploader("Chọn ảnh chụp vùng da cần khám...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh vừa up
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã tải lên', use_container_width=True)
    
    if st.button("BẮT ĐẦU CHẨN ĐOÁN"):
        # --- Xử lý 19 số Metadata tự động ---
        meta = np.zeros(19)
        meta[0] = age # Vị trí 1: Tuổi
        
        # Vị trí 2,3,4: Giới tính
        if gender == "Nữ": meta[1] = 1
        elif gender == "Nam": meta[2] = 1
        else: meta[3] = 1
        
        # Vị trí 5 -> 19: Vị trí (localization)
        loc_list = ['abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot', 
                    'genital', 'hand', 'lower extremity', 'neck', 'scalp', 'trunk', 'unknown', 'upper extremity']
        meta[4 + loc_list.index(loc)] = 1
        
        # --- Xử lý Ảnh ---
        img_array = np.array(image.convert('RGB'))
        img_resized = cv2.resize(img_array, (224, 224))
        img_final = img_resized / 255.0
        img_final = np.expand_dims(img_final, axis=0)
        
        # --- AI Dự đoán ---
        prediction = model.predict([img_final, np.array([meta])])
        classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        result_idx = np.argmax(prediction)
        
        # --- Hiển thị kết quả ---
        st.success(f"Kết quả dự đoán: **{classes[result_idx].upper()}**")
        st.info(f"Độ tin cậy: {prediction[0][result_idx]*100:.2f}%")
        # --- Hiển thị kết quả (Code cũ của ông) ---
        st.success(f"Kết quả dự đoán: **{classes[result_idx].upper()}**")
        st.info(f"Độ tin cậy: {prediction[0][result_idx]*100:.2f}%")

        # --- CHÈN THÊM ĐOẠN NÀY ĐỂ VẼ BIỂU ĐỒ ---
        st.write("---")
        st.subheader("📊 Phân tích xác suất chi tiết:")
        
        # Tạo bảng dữ liệu để vẽ biểu đồ
        # Lưu ý: Thứ tự tên bệnh phải khớp với danh sách 'classes' của ông
        ten_benh_tieng_viet = [
            'Tiền ung thư (AKIEC)', 'Ung thư tế bào đáy (BCC)', 
            'Dày sừng lành tính (BKL)', 'U xơ da (DF)', 
            'Ung thư hắc tố (MEL)', 'Nốt ruồi (NV)', 'Mạch máu (VASC)'
        ]
        
        chart_data = pd.DataFrame({
            'Loại bệnh': ten_benh_tieng_viet,
            'Xác suất (%)': prediction[0] * 100
        })
        
        # Hiển thị biểu đồ cột
        st.bar_chart(chart_data.set_index('Loại bệnh'))