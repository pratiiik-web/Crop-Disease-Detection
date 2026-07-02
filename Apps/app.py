import streamlit as st
import numpy as np
from PIL import Image
import json
import os
import tflite_runtime.interpreter as tflite

# Paths
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'crop_disease_model.tflite')
CLASS_PATH = os.path.join(BASE_DIR, 'Models', 'class_names.json')

# Disease information for all 38 classes
DISEASE_INFO = {
    'Apple___Apple_scab':
        ('🍎 Apple Scab', 'Caused by Venturia inaequalis fungus. Creates dark, scabby lesions on leaves and fruit.',
         'Remove infected leaves. Apply copper-based or sulfur fungicide early in the season.'),
    'Apple___Black_rot':
        ('🍎 Apple Black Rot', 'Caused by Botryosphaeria obtusa. Creates brown rotting spots on fruit and leaves.',
         'Prune dead wood. Apply fungicide during bloom. Remove mummified fruit.'),
    'Apple___Cedar_apple_rust':
        ('🍎 Cedar Apple Rust', 'Caused by Gymnosporangium juniperi-virginianae. Orange spots on leaves.',
         'Apply fungicide at bud break. Remove nearby cedar/juniper trees if possible.'),
    'Apple___healthy':
        ('🍎 Healthy Apple', 'Your apple plant looks healthy!', 'Maintain regular watering and fertilization.'),
    'Blueberry___healthy':
        ('🫐 Healthy Blueberry', 'Your blueberry plant looks healthy!', 'Maintain soil pH between 4.5–5.5.'),
    'Cherry_(including_sour)___Powdery_mildew':
        ('🍒 Cherry Powdery Mildew', 'Caused by Podosphaera clandestina. White powdery coating on leaves.',
         'Apply sulfur or potassium bicarbonate fungicide. Improve air circulation.'),
    'Cherry_(including_sour)___healthy':
        ('🍒 Healthy Cherry', 'Your cherry plant looks healthy!', 'Ensure good drainage and sunlight.'),
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot':
        ('🌽 Corn Gray Leaf Spot', 'Caused by Cercospora zeae-maydis. Gray rectangular lesions on leaves.',
         'Use resistant varieties. Apply fungicide at early tasseling stage.'),
    'Corn_(maize)___Common_rust_':
        ('🌽 Corn Common Rust', 'Caused by Puccinia sorghi. Orange-brown pustules on both leaf surfaces.',
         'Plant resistant hybrids. Apply fungicide if severe infection early in season.'),
    'Corn_(maize)___Northern_Leaf_Blight':
        ('🌽 Corn Northern Leaf Blight', 'Caused by Exserohilum turcicum. Long elliptical gray-green lesions.',
         'Use resistant varieties. Apply fungicide before tasseling if needed.'),
    'Corn_(maize)___healthy':
        ('🌽 Healthy Corn', 'Your corn plant looks healthy!', 'Maintain proper spacing and nitrogen levels.'),
    'Grape___Black_rot':
        ('🍇 Grape Black Rot', 'Caused by Guignardia bidwellii. Brown lesions on leaves, black shriveled fruit.',
         'Remove infected material. Apply fungicide from bud break through fruit set.'),
    'Grape___Esca_(Black_Measles)':
        ('🍇 Grape Black Measles', 'Caused by fungal complex in wood. Tiger-stripe pattern on leaves.',
         'No cure available. Remove infected vines. Avoid large pruning wounds.'),
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)':
        ('🍇 Grape Leaf Blight', 'Caused by Pseudocercospora vitis. Dark brown spots with yellow halos.',
         'Apply copper-based fungicide. Ensure good air circulation through canopy.'),
    'Grape___healthy':
        ('🍇 Healthy Grape', 'Your grape vine looks healthy!', 'Prune regularly and monitor for pests.'),
    'Orange___Haunglongbing_(Citrus_greening)':
        ('🍊 Citrus Greening (HLB)', 'Caused by Candidatus Liberibacter bacteria spread by psyllid insects. No cure.',
         'Remove infected trees immediately. Control psyllid insects. Plant certified disease-free nursery stock.'),
    'Peach___Bacterial_spot':
        ('🍑 Peach Bacterial Spot', 'Caused by Xanthomonas arboricola. Water-soaked spots on leaves and fruit.',
         'Apply copper bactericide during dormancy. Use resistant varieties.'),
    'Peach___healthy':
        ('🍑 Healthy Peach', 'Your peach plant looks healthy!', 'Thin fruit early and maintain good drainage.'),
    'Pepper,_bell___Bacterial_spot':
        ('🫑 Pepper Bacterial Spot', 'Caused by Xanthomonas campestris. Dark water-soaked spots on leaves.',
         'Use certified disease-free seeds. Apply copper bactericide. Avoid overhead irrigation.'),
    'Pepper,_bell___healthy':
        ('🫑 Healthy Pepper', 'Your pepper plant looks healthy!', 'Water consistently and avoid wetting leaves.'),
    'Potato___Early_blight':
        ('🥔 Potato Early Blight', 'Caused by Alternaria solani. Dark brown spots with concentric rings.',
         'Apply chlorothalonil or mancozeb fungicide. Remove infected lower leaves.'),
    'Potato___Late_blight':
        ('🥔 Potato Late Blight', 'Caused by Phytophthora infestans — same pathogen that caused Irish Famine.',
         'Apply fungicide preventively. Destroy infected plants immediately. Do not compost.'),
    'Potato___healthy':
        ('🥔 Healthy Potato', 'Your potato plant looks healthy!', 'Hill soil around plants and monitor for pests.'),
    'Raspberry___healthy':
        ('🍓 Healthy Raspberry', 'Your raspberry plant looks healthy!', 'Prune old canes after fruiting.'),
    'Soybean___healthy':
        ('🫘 Healthy Soybean', 'Your soybean plant looks healthy!', 'Monitor for aphids and soybean cyst nematode.'),
    'Squash___Powdery_mildew':
        ('🎃 Squash Powdery Mildew', 'Caused by Podosphaera xanthii. White powdery coating on leaf surfaces.',
         'Apply potassium bicarbonate or neem oil. Improve air circulation. Avoid wetting leaves.'),
    'Strawberry___Leaf_scorch':
        ('🍓 Strawberry Leaf Scorch', 'Caused by Diplocarpon earliana. Purple spots with scorched appearance.',
         'Remove infected leaves. Apply fungicide. Use drip irrigation to keep leaves dry.'),
    'Strawberry___healthy':
        ('🍓 Healthy Strawberry', 'Your strawberry plant looks healthy!', 'Renew beds every 3-4 years.'),
    'Tomato___Bacterial_spot':
        ('🍅 Tomato Bacterial Spot', 'Caused by Xanthomonas vesicatoria. Small dark water-soaked spots.',
         'Use copper bactericide. Avoid overhead irrigation. Use disease-free transplants.'),
    'Tomato___Early_blight':
        ('🍅 Tomato Early Blight', 'Caused by Alternaria solani. Dark spots with concentric target rings.',
         'Remove lower infected leaves. Apply mancozeb or chlorothalonil fungicide.'),
    'Tomato___Late_blight':
        ('🍅 Tomato Late Blight', 'Caused by Phytophthora infestans. Dark water-soaked lesions on leaves.',
         'Apply fungicide preventively in cool wet weather. Destroy infected plants immediately.'),
    'Tomato___Leaf_Mold':
        ('🍅 Tomato Leaf Mold', 'Caused by Passalora fulva. Yellow patches above, olive-green mold below.',
         'Reduce humidity. Improve ventilation. Apply fungicide if severe.'),
    'Tomato___Septoria_leaf_spot':
        ('🍅 Tomato Septoria Leaf Spot', 'Caused by Septoria lycopersici. Small circular spots with dark borders.',
         'Remove infected leaves. Apply fungicide. Mulch to prevent soil splash.'),
    'Tomato___Spider_mites Two-spotted_spider_mite':
        ('🍅 Tomato Spider Mites', 'Caused by Tetranychus urticae. Tiny mites causing yellow stippling on leaves.',
         'Apply neem oil or insecticidal soap. Increase humidity. Introduce predatory mites.'),
    'Tomato___Target_Spot':
        ('🍅 Tomato Target Spot', 'Caused by Corynespora cassiicola. Brown spots with concentric rings.',
         'Apply fungicide. Remove infected plant debris. Ensure good air circulation.'),
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus':
        ('🍅 Tomato Yellow Leaf Curl Virus', 'Spread by whiteflies. Causes upward curling and yellowing of leaves.',
         'Control whitefly population. Use reflective mulch. Remove infected plants.'),
    'Tomato___Tomato_mosaic_virus':
        ('🍅 Tomato Mosaic Virus', 'Spread by contact and seed. Causes mottled yellow-green mosaic pattern.',
         'Remove infected plants. Disinfect tools. Use virus-resistant varieties.'),
    'Tomato___healthy':
        ('🍅 Healthy Tomato', 'Your tomato plant looks healthy!', 'Water consistently at soil level.'),
}

@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    with open(CLASS_PATH) as f:
        class_names = json.load(f)
    return interpreter, class_names

interpreter, class_names = load_model()

# App UI
st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿", layout="centered")
st.title("🌿 Crop Disease Detector")
st.markdown("Upload a photo of a plant leaf to instantly detect disease and get treatment advice.")

st.info("💡 **Tip:** Use a clear, close-up photo of a single leaf with good lighting for best results. Supported crops: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img       = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, 0)

    # Predict
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_class = class_names[np.argmax(predictions)]
    confidence      = np.max(predictions) * 100

    st.divider()

    # Get disease info
    info = DISEASE_INFO.get(predicted_class)

    if info:
        display_name, description, treatment = info
        is_healthy = 'healthy' in predicted_class.lower()

        if is_healthy:
            st.success(f"✅ **{display_name}** — {confidence:.1f}% confidence")
        else:
            st.error(f"🔴 **{display_name}** — {confidence:.1f}% confidence")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📋 About this disease:**")
            st.write(description)
        with col2:
            st.markdown("**💊 Recommended treatment:**")
            st.write(treatment)
    else:
        plant, disease = predicted_class.split('___')
        st.warning(f"Detected: {plant.replace('_',' ')} — {disease.replace('_',' ')} ({confidence:.1f}%)")

    # Confidence warning
    if confidence < 50:
        st.warning("⚠️ Low confidence — try uploading a clearer, closer photo of the leaf.")

    st.divider()
    st.markdown("**Top 3 predictions:**")
    top_3_idx = np.argsort(predictions)[-3:][::-1]
    for idx in top_3_idx:
        cls  = class_names[idx].replace('___', ' → ').replace('_', ' ')
        prob = predictions[idx] * 100
        st.write(f"{cls}: {prob:.1f}%")
        st.progress(float(predictions[idx]))