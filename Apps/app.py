import streamlit as st
import numpy as np
from PIL import Image
import json
import tflite_runtime.interpreter as tflite

@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path='Models/crop_disease_model.tflite')
    interpreter.allocate_tensors()
    with open('Models/class_names.json') as f:
        class_names = json.load(f)
    return interpreter, class_names

interpreter, class_names = load_model()

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿")
st.title("🌿 Crop Disease Detector")
st.markdown("Upload a photo of a plant leaf to detect disease.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, 0)

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.divider()
    plant, disease = predicted_class.split('___')
    plant   = plant.replace('_', ' ')
    disease = disease.replace('_', ' ')

    if 'healthy' in disease.lower():
        st.success(f"✅ **{plant}** — Healthy ({confidence:.1f}% confidence)")
    else:
        st.error(f"🔴 **{plant}** — {disease} ({confidence:.1f}% confidence)")

    st.markdown("**Top 3 predictions:**")
    top_3_idx = np.argsort(predictions)[-3:][::-1]
    for idx in top_3_idx:
        cls  = class_names[idx].replace('___', ' - ').replace('_', ' ')
        prob = predictions[idx] * 100
        st.write(f"{cls}: {prob:.1f}%")
        st.progress(float(predictions[idx]))