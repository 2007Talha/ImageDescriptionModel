# Image Description Model

A Streamlit application that generates a short natural-language description for an uploaded image. The app uses a TensorFlow image-captioning model trained on the Flickr8k dataset.

## Features

- Upload a JPG, JPEG, or PNG image.
- Preprocess the image to the model's expected `224 x 224` RGB input.
- Generate a caption with the saved TensorFlow model.
- Display the uploaded image and the predicted caption in a browser-based Streamlit UI.

## Requirements

- Python 3.9 or newer
- TensorFlow
- Streamlit
- A CPU-compatible TensorFlow installation, or a configured GPU for faster inference

The saved vocabulary are already included in this repository:

- `vocab.pkl` - vocabulary used by the text vectorizer

## Installation

Create and activate a virtual environment, then install the dependencies:

### Windows PowerShell

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the application

From the project directory, with the virtual environment activated:

```bash
streamlit run app.py
```

Streamlit will print a local URL, normally `http://localhost:8501`. Open that URL in a browser, upload an image, and wait for the generated description.

## Project structure

```text
.
├── app.py                         # Streamlit inference application
├── image_description_model.keras  # Saved TensorFlow model
├── vocab.pkl                      # Saved model vocabulary
├── ImageDescriptionModel.ipynb    # Training and experimentation notebook
├── imagedescriptionmodel.py       # Python version of the training workflow
├── requirements.txt               # Python dependencies
└── sampleimagetotest/             # Sample images for local testing
```

## How caption generation works

1. The uploaded image is converted to RGB, resized to `224 x 224`, and normalized to values between `0` and `1`.
2. The saved model predicts the next word from the current partial caption.
3. Generation begins with `startseq` and stops at `endseq` or after the maximum sequence length is reached.
4. The generated tokens are converted back to words using `vocab.pkl`.

The model and vocabulary are cached after the first load so later predictions in the same Streamlit session start faster.

## Retrain or experiment

The training workflow is documented in `ImageDescriptionModel.ipynb` and `imagedescriptionmodel.py`. It downloads the Flickr8k dataset with `kagglehub`, preprocesses the captions, trains the model, and saves:

```text
image_description_model.keras
vocab.pkl
```

Run the notebook or script in an environment with the training dependencies installed. Retraining may require substantial memory and is faster with a compatible GPU.

## Troubleshooting

### Model assets cannot be loaded

Make sure `image_description_model.keras` and `vocab.pkl` are in the same directory as `app.py`, then start Streamlit from that directory.

### TensorFlow installation fails

Confirm that your Python version is supported by the TensorFlow release being installed. Creating a fresh virtual environment usually avoids conflicts with existing packages.

### Captions are inaccurate

The model is a small educational image-captioning model trained on Flickr8k, so descriptions may be incomplete or incorrect for images unlike its training data.
