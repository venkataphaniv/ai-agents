import os
import ollama as ol
import requests as r



q = 'What is the capital of France? How big is the city? How much is the total population? How many expatriates live in the capital?'

# Initialize the model
# model = ol.Model('gemma3:4b')

# Run inference with a prompt
# res = model.run(q)
# print(res)


oh = os.getenv('OLLAMA_HOST', 'localhost')
url = f"http://{oh}:11434/api/generate"

data = {
    "model": "gemma4",
    "prompt": q,
    "stream": False
}

res = r.post(url, json=data)
print(res.json())
