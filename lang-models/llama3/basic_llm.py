import ollama as ol


# Let's connect to the model and ask a question
try:
    print("\nConnecting to Ollama...using mistral model to ask for:\n")
    print(f'How can I write a simple Python function to add two numbers?\n')

    q = 'How can I write a simple Python function to add two numbers?'

    # Make sure 'mistral' is downloaded (ollama pull mistral)
    res = ol.chat(
        model='mistral',
        messages=[
            {'role': 'user', 'content': q }
        ]
    )

    # Print out the assistant's response
    print("\n--- AI Assistant Response ---")
    print(res['message']['content'])
    print("-----------------------------\n")

    print('Ask a question\n')
    q = input(f"Enter your question: ")
    res = ol.chat(
        model='mistral',
        messages=[
            {'role': 'user', 'content': q }
        ]
    )

    # Print out the assistant's response
    print("\n--- AI Assistant Response ---")
    print(res['message']['content'])
    print("-----------------------------\n")

except Exception as e:
    print(f"\n[Error] Could not connect to Ollama.")
    print(f"Details: {e}")
    print("Please make sure Ollama is running and you have pulled the 'mistral' model.\n")

