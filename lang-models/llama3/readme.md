# Run a Powerful LLM Locally

We can run powerful, multi-billion-parameter large language models like Llama 3.1 and Mistral on your own laptop. Onr of such an LLM is Ollama

## What is Ollama

Ollama is a personal, local workshop for LLMs.

In a typical workshop, you have your tools, workbench, and raw materials.

Ollama is the software that bundles everything you need into one tidy package:

    - The Model Weights: The brain of the LLM (e.g., Llama 3.1).
    - The Configuration: All the settings that tell the model how to behave.
    - The Engine: The code needed to actually run the model efficiently on your specific hardware (Mac, Win, or Linux machine).

Ollama is an open-source tool that manages all this complexity for you. It hides the difficult setup and gives you a simple command-line interface (and an API) to download, manage, and interact with a huge library of open-source models

## Install Ollama

Just download and install from <https://ollama.com>

## Pull Your First Model

Now, let’s download a model. We’ll use Mistral, which is a fantastic, high-performing model that’s a great size for most laptops. In your terminal of your VS Code (or anywhere you are working), just type:

    ollama pull mistral

## Bring Your Model Into Your Code

When you run Ollama, it starts a lightweight server in the background on your machine. All we need to do is have our code talk to that server. The Ollama team has made this incredibly simple with an official Python library.

First, you’ll need to install the client library using pip.

    pip install ollama

## Start writing your scripts

Let’s write a simple Python script (you can save this as app.py). This script will connect to your local Ollama server, send a prompt to the Mistral model, and print the response (make sure your Ollama application is running in the background!)

    import ollama

    # Let's connect to the model and ask a question
    try:
        # Make sure 'mistral' is downloaded (ollama pull mistral)
        response = ollama.chat(
            model='mistral',
            messages=[
                {'role': 'user', 'content': 'How can I write a simple Python function to add two numbers?'}
            ]
        )

        # Print out the assistant's response
        print("\n--- AI Assistant Response ---")
        print(response['message']['content'])
        print("-----------------------------\n")

    except Exception as e:
        print(f"\n[Error] Could not connect to Ollama.")
        print(f"Details: {e}")
        print("Please make sure Ollama is running and you have pulled the 'mistral' model.\n")
