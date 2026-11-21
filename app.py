
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from flower_mapping import FLOWER_CLASSES

MODEL_PATH = "models/bloomscan_transfer_fixed_best.keras"
IMAGE_SIZE = (160, 160)


# Load model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# Page settings
st.set_page_config(
    page_title="BloomScan - Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f4fff7, #fff5fa);
}
.main-title {
    text-align: center;
    color: #176b45;
    font-size: 48px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #527765;
    font-size: 18px;
}
.result-box {
    background-color: #e5f7eb;
    padding: 22px;
    border-radius: 15px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# Website heading
st.markdown('<div class="main-title">🌸 BloomScan</div>',
            unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">AI-Powered Flower Classification System</div>',
    unsafe_allow_html=True
)

st.write("")
st.write("Upload a flower image and let BloomScan identify it.")

# Image uploader
uploaded_file = st.file_uploader(
    "Choose a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Flower Image",
        use_container_width=True
    )

    if st.button("Identify Flower 🌼", type="primary"):

        with st.spinner("BloomScan is identifying your flower..."):

            model = load_model()

            resized_image = image.resize(IMAGE_SIZE)

            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            image_array = image_array / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)[0]

            predicted_index = int(np.argmax(predictions))
            flower_name = FLOWER_CLASSES[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

        st.success("Flower Identification Completed!")

        st.markdown(
            f"""
            <div class="result-box">
                <h2>🌺 {flower_name}</h2>
                <h3>Confidence: {confidence:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Top 3 Predictions")

        top_indices = np.argsort(predictions)[-3:][::-1]

        for index in top_indices:
            name = FLOWER_CLASSES[int(index)]
            score = float(predictions[index]) * 100

            st.write(f"**{name}** — {score:.2f}%")
            st.progress(float(predictions[index]))

else:
    st.info("Please upload a flower image to get started.")

st.write("---")
st.caption("BloomScan | Flower Classification using CNN")
