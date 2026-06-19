import os
import ollama as ol
import requests as r



q = """
You are a professional English (en) to Spanish (es) translator. Your goal is to accurately convey the meaning and nuances of the original English text while adhering to Spanish grammar, vocabulary, and cultural sensitivities.
Produce only the Spanish translation, without any additional explanations or commentary. Please translate the following English text into Telugu (te-IN). Please give sufficient spacesbetween characters when printing the output to make it readable:


Hello, how are you?
What's your name?
Where are you from?
What do you do for a living?
Do you have any hobbies or interests?
Do you believe in god?
"""

oh = os.getenv('OLLAMA_HOST', 'localhost')
url = f"http://{oh}:11434/api/generate"

data = {
    "model": "translategemma",
    "prompt": q,
    "stream": False
}

res = r.post(url, json=data)
print(res.json()['response'])
