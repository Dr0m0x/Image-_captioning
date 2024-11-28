import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from PIL import Image
import os

# Load the pre-trained InceptionV3 model (without the top classification layer)
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# Function to extract image features using InceptionV3
def extract_features(img_path):
    try:
        img = Image.open(img_path).resize((299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)

        # Extract features using the model
        features = inception_model.predict(img_array)
        return features.flatten()
    except Exception as e:
        print(f"Error extracting features for {img_path}: {e}")
        return None

# Load the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to generate captions using GPT-2
def generate_caption(features):
    try:
        input_text = (
            f"This image contains features with an average intensity of {np.mean(features):.2f}. "
            "The scene may include textures, shapes, or objects based on these characteristics."
        )
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate the caption
        output = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=True,  # Enable sampling
            temperature=0.7,  # Control randomness
            top_p=0.9,  # Nucleus sampling
            pad_token_id=tokenizer.eos_token_id,
        )
        caption = tokenizer.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        return None

# Function to process an image and generate its caption
def caption_image(img_path):
    print(f"Processing: {img_path}")
    features = extract_features(img_path)
    if features is not None:
        caption = generate_caption(features)
        return caption
    return "Failed to generate caption."

if __name__ == "__main__":
    # Path to the dataset directory
    dataset_dir = "dataset"  # Replace with your dataset directory
    if not os.path.exists(dataset_dir):
        print("Dataset directory not found.")
        exit()

    # Iterate through the images in the dataset directory
    for img_file in os.listdir(dataset_dir):
        img_path = os.path.join(dataset_dir, img_file)
        if os.path.isfile(img_path):
            caption = caption_image(img_path)
            print(f"Generated Caption for {img_path}: {caption}")
