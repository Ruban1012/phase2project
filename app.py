import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Page setup
st.set_page_config(page_title="Microplastic Detection", layout="wide")

st.title("🌊 Microplastic Detection System")

# Load model safely
try:
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
except:
    st.error("❌ model.tflite not found! Please keep it in same folder as app.py")
    st.stop()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Upload image
file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, width="stretch")

    # Preprocess
    h, w = input_details[0]['shape'][1:3]
    img = image.resize((w, h))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

    st.write("Raw Output:", float(pred))

    # Result
    if pred > 0.5:
        st.error(f"⚠️ Microplastic Detected ({pred*100:.2f}%)")
    else:
        st.success(f"💧 Clean Water ({(1-pred)*100:.2f}%)")
