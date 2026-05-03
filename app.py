import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Microplastic Detection", layout="wide")

st.title("🌊 Microplastic Detection System")

# --------------------------
# LOAD TFLITE MODEL
# --------------------------
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --------------------------
# UPLOAD IMAGE
# --------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width="stretch")

    # --------------------------
    # PREPROCESS (IMPORTANT)
    # --------------------------
    input_shape = input_details[0]['shape']
    h, w = input_shape[1], input_shape[2]

    img = image.resize((w, h))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    # Match dtype
    if input_details[0]['dtype'] == np.float32:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.uint8)

    # --------------------------
    # PREDICT
    # --------------------------
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

    st.write("Raw Output:", float(pred))

    # --------------------------
    # RESULT
    # --------------------------
    if pred > 0.5:
        st.error(f"⚠️ Microplastic ({pred*100:.2f}%)")
    else:
        st.success(f"💧 Clean Water ({(1-pred)*100:.2f}%)")
