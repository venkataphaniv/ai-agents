"""
Basic example of using Ollama API for image processing with LLaVA model
"""

import requests as r
import json as j
import base64 as b64
import os


def encode_image(image_path):
    """Convert image to base64 encoding for Ollama API"""
    with open(image_path, "rb") as img_fp:
        return b64.b64encode(img_fp.read()).decode('utf-8')


def analyze_image(image_path, prompt="Describe this image in detail"):
    """Send image and prompt to Ollama for analysis"""

    # Encode the image
    b64_img = encode_image(image_path)

    # Prepare the request
    payload = {
        "model": "llava",
        "prompt": prompt,
        "images": [b64_img],
        "stream": False
    }

    # Send request to Ollama
    res = r.post(
        "http://localhost:11434/api/generate",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if res.status_code == 200:
        return res.json()['response']
    else:
        return f"Error: {res.status_code}"


# Test the function
if __name__ == "__main__":
    img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "6-steps-thematic-analysis.png")
    description = analyze_image(img, prompt="Provide a detailed analysis of this image.")
    print(description)

