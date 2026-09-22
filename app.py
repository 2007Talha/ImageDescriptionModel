import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.layers import TextVectorization
from PIL import Image

st.set_page_config(
    page_title="AI Image description Generator",
    page_icon="📷",
    layout="centered"
    
)

VOCAB_SIZE = 10000
MAX_LENGTH = 32

@st.cache_resource
def load_assets():

    model = tf.keras.models.load_model("image_description_model.keras")
    
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
        
    vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_LENGTH
    )
    vectorizer.set_vocabulary(vocab)
    
    vocab_list = vectorizer.get_vocabulary()
    int_to_word = {i: word for i, word in enumerate(vocab_list)}
    
    return model, vectorizer, int_to_word

try:
    model, vectorizer, int_to_word = load_assets()
    tf.keras.backend.set_floatx('float32')
except Exception as e:
    st.error(f"Error loading model assets. Make sure 'image_captioning_model.keras' and 'vocab.pkl' are in the same folder! Error: {e}")
    st.stop()

def preprocess_pil_image(pil_image):
    img = pil_image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

def generate_caption(image_tensor):

    in_text = "startseq"
    for i in range(MAX_LENGTH - 1):

        sequence = vectorizer([in_text])
        sequence_input = sequence[:, :-1]
        predictions = model.predict([image_tensor, sequence_input], verbose=0)

        if len(predictions.shape) == 3:
            predicted_id = np.argmax(predictions[0, i, :])
        else:
            predicted_id = np.argmax(predictions[0])
        word = int_to_word.get(predicted_id, "")
        if word == "endseq" or word == "":
            break
        in_text += " " + word
        
    final_caption = in_text.replace("startseq", "").strip()
    return final_caption

st.title("☕ Image description Bot")
st.markdown("Upload any picture below, and our trained Encoder-Decoder model will attempt to describe what is happening inside it.")
st.markdown("---")

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Source Image", use_container_width=True)
    st.markdown("### Model Analysis")

    with st.spinner("🧠 Visualizing features and generating caption sequences..."):
        processed_image = preprocess_pil_image(image)
        predicted_caption = generate_caption(processed_image)
    st.success(f"**Predicted Caption:** {predicted_caption}")
