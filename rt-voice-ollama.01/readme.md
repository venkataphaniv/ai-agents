# Build a Real-Time Voice AI Assistant

We will build a voice AI assistant that listens to you, thinks using a powerful local AI model (Llama 3), and responds to you

## How a Voice AI Assistant Works in Real-time?

Before we write the code, let’s understand what we are building.

An AI voice assistant is essentially a loop of three distinct biological functions replicated by code:

    - The Ears (Speech-to-Text): We capture audio vibrations and translate them into text.

    - The Brain (LLM Inference): We send that text to a Large Language Model (Ollama/Llama 3) to generate a smart response.

    - The Mouth (Text-to-Speech): We convert the AI’s text response back into audio so we can hear it.

Let’s understand it practically by building a real-time voice AI assistant using Python

## Building a Real-Time Voice AI Assistant

To get it running, we are relying on three key libraries. You will need to install them via your terminal:

    pip install speechrecognition ollama pyttsx3 pyaudio

You must have the Ollama application installed on your computer and the Llama 3 model pulled (ollama pull llama3) for the brain part of our code to work.
