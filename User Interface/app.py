from flask import Flask, render_template, request, send_from_directory
from diffusers import StableDiffusionPipeline
import torch
import os
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the Stable Diffusion model
def load_model():
    model_id = "CompVis/stable-diffusion-v-1-4-original"  # Use the Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda")  # Use GPU if available
    return pipe

# Function to generate image from the prompt
def generate_image(pipe, prompt):
    # Generate image from the prompt
    image = pipe(prompt).images[0]
    return image

# Ensure the output directory exists
output_dir = "static/images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize model once when the app starts
pipe = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt')
    if prompt:
        try:
            # Generate image from the prompt
            image = generate_image(pipe, prompt)
            
            # Save image to a file
            image_path = os.path.join(output_dir, 'generated_image.png')
            image.save(image_path)

            # Send the image path to render on the webpage
            return render_template('index.html', prompt=prompt, image_path=image_path)
        except Exception as e:
            return render_template('index.html', error=str(e))
    else:
        return render_template('index.html', error="Please enter a prompt.")

@app.route('/static/images/<filename>')
def get_image(filename):
    return send_from_directory(output_dir, filename)

if __name__ == "__main__":
    app.run(debug=True)
