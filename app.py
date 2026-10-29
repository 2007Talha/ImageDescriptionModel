import io
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import TextVectorization

st.set_page_config(page_title="Image Insight Studio", page_icon="🖼️", layout="wide")

VOCAB_SIZE = 10000
MAX_LENGTH = 32
MODEL_CANDIDATES = ["image_description_model.keras", "image_captioning_model.keras"]

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 4rem;
            padding-bottom: 3rem;
        }
        .hero {
            background: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 12px;
            padding: 1.5rem 1.75rem;
            min-height: 0;
            overflow: visible;
            margin-bottom: 1.5rem;
        }
        .caption-box {
            background: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-left: 4px solid var(--primary-color);
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-top: 0.75rem;
        }
        .info-pill {
            display: inline-flex;
            align-items: center;
            background: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 6px;
            padding: 0.35rem 0.65rem;
            font-size: 0.8rem;
            color: var(--text-color);
            margin-right: 0.4rem;
            margin-bottom: 0.5rem;
        }
        div[data-testid="stFileUploader"] section {
            background: var(--secondary-background-color);
            border: 1px dashed rgba(128, 128, 128, 0.5);
            border-radius: 10px;
            padding: 0.5rem;
        }
        div[data-testid="stImage"] img {
            max-height: 560px;
            object-fit: contain;
            object-position: center;
        }
        .stButton > button {
            border-radius: 6px;
            font-weight: 600;
            padding: 0.5rem 1rem;
        }
        div[data-testid="stStatusWidget"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def find_model_file():
    for name in MODEL_CANDIDATES:
        candidate = Path.cwd() / name
        if candidate.exists():
            return candidate

    for match in Path.cwd().glob("*.keras"):
        return match
    return None


@st.cache_resource
def load_assets():
    model_path = find_model_file()
    if model_path is None:
        raise FileNotFoundError(
            "No saved model file found. Place a .keras model file in the project folder or update the filename."
        )

    model = tf.keras.models.load_model(model_path)

    vocab_path = Path.cwd() / "vocab.pkl"
    if not vocab_path.exists():
        raise FileNotFoundError("vocab.pkl was not found in the project folder.")

    with vocab_path.open("rb") as file:
        vocab = pickle.load(file)

    vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_LENGTH,
    )
    vectorizer.set_vocabulary(vocab)

    vocab_list = vectorizer.get_vocabulary()
    int_to_word = {index: word for index, word in enumerate(vocab_list)}

    image_encoder = tf.keras.Model(
        inputs=model.inputs[0],
        outputs=model.get_layer("dense").output,
        name="image_encoder",
    )

    image_features_input = tf.keras.Input(shape=(256,), name="image_features")
    text_input = tf.keras.Input(shape=(MAX_LENGTH - 1,), dtype=tf.int32, name="text_input")
    image_features_sequence = model.get_layer("repeat_vector")(image_features_input)
    text_features = model.get_layer("embedding")(text_input)
    text_features = model.get_layer("lstm")(text_features)
    merged_features = model.get_layer("add")([image_features_sequence, text_features])
    decoder_features = model.get_layer("time_distributed")(merged_features)
    decoder_output = model.get_layer("final_word_predictor")(decoder_features)
    decoder = tf.keras.Model(
        inputs=[image_features_input, text_input],
        outputs=decoder_output,
        name="caption_decoder",
    )

    return model, vectorizer, int_to_word, image_encoder, decoder


def preprocess_pil_image(pil_image):
    img = pil_image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(img_array, dtype=tf.float32)


def generate_caption(image_tensor):
    image_features = image_encoder(image_tensor, training=False)
    in_text = "startseq"
    for _ in range(MAX_LENGTH - 1):
        sequence = vectorizer([in_text])
        sequence_input = sequence[:, :-1]
        predictions = decoder([image_features, sequence_input], training=False)
        predicted_id = np.argmax(predictions[0, -1, :]) if len(predictions.shape) == 3 else np.argmax(predictions[0])
        word = int_to_word.get(predicted_id, "")
        if word in ("endseq", ""):
            break
        in_text += " " + word
    return in_text.replace("startseq", "").strip()


@st.cache_data(show_spinner=False, max_entries=32)
def generate_caption_cached(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processed_image = preprocess_pil_image(image)
    return generate_caption(processed_image)


st.markdown(
    """
    <div class="hero">
        <h1 style="margin:0;">Image Insight Studio</h1>
        <p style="margin:0.55rem 0 0; color:var(--text-color); opacity:0.78; font-size:1rem;">
            Upload a photo and let the trained encoder-decoder model describe what it sees.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div>
        <span class="info-pill">224×224 RGB input</span>
        <span class="info-pill">TensorFlow captioning model</span>
        <span class="info-pill">Local inference</span>
    </div>
    """,
    unsafe_allow_html=True,
)

model_status = st.empty()
model_ready = False
model_error = None

try:
    model, vectorizer, int_to_word, image_encoder, decoder = load_assets()
    tf.keras.backend.set_floatx("float32")
    model_ready = True
    model_status.success("✅ Model loaded and ready for inference.")
except Exception as exc:
    model_error = exc
    model_status.warning(
        "⚠️ Model assets could not be loaded. Add the saved `.keras` model and `vocab.pkl` in this folder to enable captions."
    )

left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    st.subheader("Upload an image")
    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG file",
        type=["jpg", "jpeg", "png"],
        help="Upload a photo to generate a caption.",
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", width="stretch")

with right_col:
    st.subheader("Try a sample")
    sample_dir = Path.cwd() / "sampleimagetotest"
    sample_images = sorted(sample_dir.glob("*.*")) if sample_dir.exists() else []
    if sample_images:
        sample_name = st.selectbox(
            "Select a sample image",
            ["-- Select a sample --"] + [img.name for img in sample_images],
            label_visibility="collapsed",
        )
        if sample_name != "-- Select a sample --":
            selected_image = Image.open(sample_dir / sample_name)
            st.image(selected_image, caption=f"Sample: {sample_name}", width="stretch")
            sample_for_inference = selected_image
        else:
            sample_for_inference = None
    else:
        sample_for_inference = None

    if uploaded_file is not None:
        source_image = Image.open(uploaded_file)
    else:
        source_image = sample_for_inference

    if source_image is not None:
        st.subheader("Caption")
        with st.spinner("Analyzing the scene and generating a description..."):
            if model_ready:
                image_buffer = io.BytesIO()
                source_image.convert("RGB").save(image_buffer, format="PNG")
                predicted_caption = generate_caption_cached(image_buffer.getvalue())
                st.markdown(
                    f'<div class="caption-box"><p style="margin:0; font-size:1.05rem; line-height:1.65;">{predicted_caption}</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("The app is ready, but a trained model is not available yet. Add the saved model file to start generating captions.")

if model_error is not None and uploaded_file is None and not sample_images:
    st.caption("Tip: run the notebook once to train and save the model, then restart this app.")
