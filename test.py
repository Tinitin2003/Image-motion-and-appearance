import numpy as np
import tensorflow as tf
import pickle
import gradio as gr
import spacy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from nltk.corpus import wordnet as wn
import base64
from pathlib import Path
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

max_length = 74  # Maximum length of the caption

import pyttsx3  # Add the text-to-speech library

# Initialize the TTS engine
engine = pyttsx3.init()
# Set slower speech rate (default is ~200 wpm)
engine.setProperty('rate', 160)  # Lower value = slower speech

# Optional: set voice (male/female), language
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Change index as needed

def speak_text(text):
    """Convert text to speech and speak it."""
    engine.say(text)
    engine.runAndWait()


with open("Token File/word_index_flikr.pkl", "rb") as f:
    index_word = pickle.load(f)

with open("Token File/index_word.pkl", "rb") as f:
    index_word_2 = pickle.load(f)


def idx_to_word(integer, index_word_dict):
    return index_word_dict.get(integer, None)


word_to_index = {word: idx for idx, word in index_word.items()}

word_to_index_2 = {word: idx for idx, word in index_word_2.items()}

def words_to_sequence(sentence, word_to_index):
    return [word_to_index.get(word, 0) for word in sentence.split()]  # 0 = unknown/padding



# Load DenseNet201 model for feature extraction
model = DenseNet201()
fe = Model(inputs=model.input, outputs=model.layers[-2].output)

# Function to extract features from an image
def extract_features(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))  # Resize image to DenseNet input size
    img = img_to_array(img)
    img = img/255.
    img = np.expand_dims(img,axis=0)
    features = fe.predict(img)
    return features


# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load both trained models
early_30 = load_model('Model/Early_Fusion_flikr30k.keras')        # Early Fusion
late_30 = load_model('Model/Late_Fusion_flikr30k.keras')  # Late Fusion
early_150 = load_model('Model/Early_Fusion_Model_150k_data.keras')        # Early Fusion
# Model name to model object mapping
model_dict = {
    "Early Fusion: Basic": early_30,
    "Late Fusion: Basic": late_30,
    "Early Fusion: Intermediate": early_150,
}

# Predefined motion words
motion_words = {
    "run", "walk", "jump", "fly", "swim", "move", "crawl", "slide",
    "throw", "catch", "push", "pull", "drag", "lift", "climb", "dive",
    "skip", "hop", "roll", "stumble", "fall", "dash", "chase"
}

def get_motion_synonyms():
    motion_synsets = wn.synset('move.v.01').closure(lambda s: s.hyponyms())
    for synset in motion_synsets:
        motion_words.update(lemma.name().replace("_", " ") for lemma in synset.lemmas())

get_motion_synonyms()

def detect_motion(text):
    doc = nlp(text)
    for token in doc:
        if token.lemma_ in motion_words and token.pos_ == "VERB":
            return "Motion Detected"
    return "No Motion Detected"

def predict_caption(image_path,selected_model_name):
    """Generates caption and detects motion."""
    caption_model = model_dict[selected_model_name]

    feature = extract_features(image_path)  
    caption = "startseq"

    # Generate caption
    for _ in range(max_length):
        seq = words_to_sequence(caption, word_to_index)
        if(selected_model_name == "Early Fusion: Advanced"):
             seq = words_to_sequence(caption, word_to_index_2)
        seq = pad_sequences([seq], maxlen=max_length, padding='post')

        yhat = caption_model.predict([feature, seq], verbose=0)
        yhat = np.argmax(yhat)  

        #word = idx_to_word(yhat, tokenizer)
        word=idx_to_word(yhat, index_word)
        if(selected_model_name == "Early Fusion: Advanced"):
            word=idx_to_word(yhat, index_word_2)
        if word is None or word == 'endseq':
            break

        caption += " " + word

    caption = caption.replace("startseq", "").strip()
    
    # Detect motion
    motion_result = detect_motion(caption)
    speak_caption("This appears to be {caption} based on visual analysis and.".format(caption=caption))
    if(motion_result == "Motion Detected"):
        speak_caption("Motion has been detected in the current frame.")
    else:
        speak_caption("No motion is currently observed in the scene.")
    return caption, motion_result


## Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🖼️ Image Motion and Appearance Detector")
    gr.Markdown("Upload an image, choose a model, and get a generated caption with motion detection.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload Image", sources=["upload"])
            model_selector = gr.Dropdown(choices=["Early Fusion: Basic", "Late Fusion: Basic", "Early Fusion: Intermediate"], value="Early Fusion: Basic", label="Choose Captioning Model")
            submit_button = gr.Button("Generate Caption", variant="primary")
        
        with gr.Column():
            caption_output = gr.Textbox(label="Image Description", lines=3, placeholder="Generated caption will appear here...")
            motion_output = gr.Textbox(label="Motion Detection Result", lines=2, placeholder="Motion detection result will appear here...")
    def update_ui(image_path, selected_model_name):
        caption, motion_result = predict_caption(image_path, selected_model_name)
        return caption, motion_result

    def speak_caption(caption):
        speak_text(caption)
        return "Speaking caption..."  # Just an indication that the caption is being spoken

    submit_button.click(
        fn=update_ui,
        inputs=[image_input, model_selector],
        outputs=[caption_output, motion_output]
    )


    gr.Markdown("### How it works:")
    gr.Markdown("1. Upload an image using the 'Upload Image' button.")
    gr.Markdown("2. Choose a captioning model.")
    gr.Markdown("3. Click 'Generate Caption' to get a description of the image.")
    gr.Markdown("4. Click 'Speak Caption' to hear the generated caption.")

demo.launch(share=True)