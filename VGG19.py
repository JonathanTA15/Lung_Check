import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the VGG19 model
model_path = 'VGG19.h5'  # Ganti dengan path sesuai lokasi model Anda
model = load_model(model_path)

# Fungsi untuk memproses gambar
def process_image(image):
    # Resize gambar menjadi ukuran yang diharapkan oleh VGG19 (misalnya 224x224 pixel)
    size = (224, 224)
    image = ImageOps.fit(image, size)
    image = image.convert('RGB')  # Pastikan gambar diubah menjadi mode RGB jika awalnya berwarna hitam putih
    
    # Normalisasi nilai pixel
    image = np.asarray(image)
    image = (image.astype('float32') / 255.0)
    
    return image

# Judul dan deskripsi aplikasi
st.title('Klasifikasi Gambar CT-Scan dengan VGG19')
st.write('Upload gambar CT-Scan untuk diklasifikasikan.')

# Widget untuk upload gambar
uploaded_file = st.file_uploader("Pilih gambar CT-Scan...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Tampilkan gambar yang diupload
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diupload', use_column_width=True)
    
    # Prediksi kelas gambar
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    
    prediction = model.predict(processed_image)
    
    classes = ['Bengin cases', 'Malignant cases', 'Normal cases']
    predicted_class = classes[np.argmax(prediction)]
    
    st.write(f'Kelas Prediksi: {predicted_class}')
