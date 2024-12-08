import os
import torch
from flask import Flask, render_template, request, send_file
from PIL import Image
import random

app = Flask(__name__)

# Placeholder AI image generation function (simulate with random images for now)
def generate_image(prompt):
    # Placeholder: Creating a random image as a stand-in for the AI model output
    img = Image.new('RGB', (512, 512), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    img.save("generated_image.png")
    return "generated_image.png"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    if prompt:
        image_path = generate_image(prompt)  # Call the AI model to generate the image
        return send_file(image_path, mimetype='image/png')
    return "Please provide a prompt."

if __name__ == "__main__":
    app.run(debug=True)
