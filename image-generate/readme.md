# Get up and running with AI image generation

Run FLUX.1, Stable Diffusion 3.5, Stable Diffusion 1.5, and other models, locally.

## Insructions

### Install from PyPI

pip install ollamadiffuser

### Pull and run a model (4-command setup)

ollamadiffuser pull flux.1-schnell

ollamadiffuser run flux.1-schnell

### Generate via API

curl -X POST <http://localhost:8000/api/generate> \
-H "Content-Type: application/json" \
-d '{"prompt": "A beautiful sunset"}' \
--output image.png
