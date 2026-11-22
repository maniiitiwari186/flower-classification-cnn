
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from flower_mapping import FLOWER_CLASSES

MODEL_PATH = "models/bloomscan_transfer_fixed_best.keras"
IMAGE_SIZE = (160, 160)


# Flower descriptions
FLOWER_DESCRIPTIONS = {
    "Rose": "Rose is a popular flowering plant known for its beautiful petals and pleasant fragrance. It comes in different colours and is commonly grown in gardens.",
    
    "Sunflower": "Sunflower is a bright yellow flower known for its large flower head. It is famous for turning towards sunlight and producing edible seeds.",
    
    "Daisy": "Daisy is a simple and attractive flower with a central yellow disc surrounded by white or colourful petals. It is commonly found in gardens and meadows.",
    
    "Tulip": "Tulip is a colourful spring flower known for its cup-shaped petals. It comes in many colours and is widely grown in gardens.",
    
    "Lotus": "Lotus is an aquatic flower that grows in ponds and lakes. It is known for its beautiful petals and large floating leaves.",
    
    "Dandelion": "Dandelion is a bright yellow flowering plant. After flowering, it forms a fluffy seed head that can spread its seeds through the wind.",
    
    "Lily": "Lily is an elegant flowering plant known for its large, attractive petals and pleasant fragrance. It comes in several colours and varieties.",
    
    "Orchid": "Orchid is an exotic flowering plant known for its unique shapes and beautiful patterns. It is commonly grown as an ornamental plant.",
    
    "Daffodil": "Daffodil is a cheerful spring flower, usually yellow or white, with a trumpet-shaped centre. It is commonly associated with the arrival of spring.",
    
    "Crocus": "Crocus is a small flowering plant that blooms in early spring. Its flowers are commonly purple, yellow or white."
}


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
.description-box {
    background-color: #fff5fa;
    padding: 18px;
    border-radius: 15px;
    border-left: 5px solid #176b45;
}
</style>
""", unsafe_allow_html=True)


# Website heading
st.markdown(
    '<div class="main-title">🌸 BloomScan</div>',
    unsafe_allow_html=True
)

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

            flower_description = FLOWER_DESCRIPTIONS.get(
                flower_name,
                "No description available for this flower."
            )

        st.success("Flower Identification Completed!")

        # Prediction result
        st.markdown(
            f"""
            <div class="result-box">
                <h2>🌺 {flower_name}</h2>
                <h3>Confidence: {confidence:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Flower description
        st.write("")
        st.subheader(f"About {flower_name}")

        st.markdown(
            f"""
            <div class="description-box">
                {flower_description}
            </div>
            """,
            unsafe_allow_html=True
        )


else:
    st.info("Please upload a flower image to get started.")


st.write("---")
st.caption("BloomScan | Flower Classification using CNN")
