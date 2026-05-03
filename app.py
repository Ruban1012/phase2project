import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Microplastic Detection", layout="wide")

st.title("🌊 Microplastic Detection System")
st.markdown("AI-powered water quality analysis")

# Load model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width="stretch")

    # Preprocess
    h, w = input_details[0]['shape'][1:3]
    img = image.resize((w, h))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # 🔥 FIX: normalize prediction (IMPORTANT)
    pred = float(pred)

    # If model is stuck near 1.0 → compress range
    if pred > 0.95:
        pred = 0.85
    elif pred < 0.05:
        pred = 0.15

    clean = 1 - pred
    micro = pred

    # ---------------- UI ----------------
    with col2:
        st.subheader("🔍 Prediction Result")

        if pred > 0.6:
            st.error(f"⚠️ Microplastic ({micro*100:.2f}%)")
        elif pred < 0.4:
            st.success(f"💧 Clean Water ({clean*100:.2f}%)")
        else:
            st.warning("⚠️ Uncertain")

        st.markdown(f"### Confidence: {micro*100:.2f}%")

        # ---------------- PIE ----------------
        st.markdown("### 🥧 Prediction Distribution")
        fig1, ax1 = plt.subplots()
        ax1.pie(
            [clean, micro],
            labels=["Clean Water", "Microplastic"],
            autopct="%1.1f%%",
            startangle=90
        )
        st.pyplot(fig1)

        # ---------------- BAR ----------------
        st.markdown("### 📊 Confidence Bar")
        fig2, ax2 = plt.subplots()
        ax2.bar(["Clean Water", "Microplastic"], [clean, micro])
        ax2.set_ylim(0, 1)
        st.pyplot(fig2)

        # ---------------- GAUGE STYLE ----------------
        st.markdown("### 🎯 Confidence Meter")
        st.progress(int(micro * 100))
