import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

IMG_SIZE = 224

@st.cache_resource
def load_model():
    with open('../Models/class_names.json') as f:
        class_names = json.load(f)

    num_classes = len(class_names)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    model.load_weights('../Models/crop_disease_model.h5', by_name=True, skip_mismatch=True)

    return model, class_names

model, class_names = load_model()

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿")
st.title("🌿 Crop Disease Detector")
st.markdown("Upload a photo of a plant leaf to detect disease.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    st.divider()

    plant, disease = predicted_class.split('___')
    plant = plant.replace('_', ' ')
    disease = disease.replace('_', ' ')

    if 'healthy' in disease.lower():
        st.success(f"✅ **{plant}** — Healthy ({confidence:.1f}% confidence)")
    else:
        st.error(f"🔴 **{plant}** — {disease} ({confidence:.1f}% confidence)")

    st.markdown("**Top 3 predictions:**")
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    for idx in top_3_idx:
        cls = class_names[idx].replace('___', ' - ').replace('_', ' ')
        prob = predictions[0][idx] * 100
        st.write(f"{cls}: {prob:.1f}%")
        st.progress(float(predictions[0][idx]))