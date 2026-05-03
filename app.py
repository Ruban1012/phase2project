import streamlit as st
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

st.set_page_config(page_title="Microplastic Detection", layout="wide")

st.title("🌊 Microplastic Detection System")

# Load model
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width="stretch")

    # preprocess
    h, w = input_details[0]['shape'][1:3]
    img = image.resize((w, h))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    if input_details[0]['dtype'] == np.float32:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.uint8)

    # predict
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

    st.write("Raw Output:", float(pred))

    if pred > 0.5:
        st.error("⚠️ Microplastic")
    else:
        st.success("💧 Clean Water")
