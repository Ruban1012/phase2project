import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Microplastic Detection", layout="wide")

st.title("🌊 Microplastic Detection System")
st.markdown("AI-powered water quality analysis")

# --------------------------
# LOAD MODEL
# --------------------------
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --------------------------
# UPLOAD
# --------------------------
file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width="stretch")

    # --------------------------
    # PREPROCESS
    # --------------------------
    h, w = input_details[0]['shape'][1:3]
    img = image.resize((w, h))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # --------------------------
    # PREDICT
    # --------------------------
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # 🔥 FIX: clamp values (avoid weird 1.0 bug)
    pred = float(np.clip(pred, 0.0, 1.0))

    clean_prob = 1 - pred
    micro_prob = pred

    # --------------------------
    # RESULT UI
    # --------------------------
    with col2:
        st.subheader("🔍 Prediction Result")

        if pred > 0.6:
            st.error(f"⚠️ Microplastic Detected ({micro_prob*100:.2f}%)")
        elif pred < 0.4:
            st.success(f"💧 Clean Water ({clean_prob*100:.2f}%)")
        else:
            st.warning("⚠️ Uncertain Prediction")

        st.markdown(f"### Confidence: {micro_prob*100:.2f}%")

        # --------------------------
        # BAR CHART
        # --------------------------
        st.markdown("### 📊 Confidence Bar Chart")

        fig, ax = plt.subplots()
        ax.bar(["Clean Water", "Microplastic"], [clean_prob, micro_prob])
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        # --------------------------
        # PIE CHART
        # --------------------------
        st.markdown("### 🥧 Prediction Distribution")

        fig2, ax2 = plt.subplots()
        ax2.pie(
            [clean_prob, micro_prob],
            labels=["Clean Water", "Microplastic"],
            autopct="%1.1f%%"
        )
        st.pyplot(fig2)
