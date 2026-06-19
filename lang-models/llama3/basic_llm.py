import ollama as ol


def query_ollama(m: str, q: str):
    # Let's connect to the model and ask a question
    try:
        # Make sure 'mistral' is downloaded (ollama pull mistral)
        res = ol.chat(
            model=m,
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
        print(f"Please make sure Ollama is running and you have pulled the '{m}' model.\n")


def get_query(m: str):
    q = 'How can I write a simple Python function to add two numbers?'
    print(f"\nConnecting to Ollama...using {m} model to ask for:\ne.g. {q}")
    print(f'Ask a question\n')
    b = True
    while b:
        q = input(f"Enter your question: ")
        ql = q.strip().lower()
        if ql in ["bye", "exit", "quit", "stop", "end", "close", "goodbye", "see you later", "talk to you later", "catch you later", "farewell", "adios", "sayonara", "au revoir", "arrivederci", "tschüss", "do svidaniya"]:
            print("Exiting... Goodbye!")
            b = False
        elif ql == "":
            print("No question entered. try again....")
        else:
            query_ollama(m, q)


if __name__ == "__main__":
    print("This is a basic test of the Ollama API connection.\nPlease run the multi-agent system to see it in action.\n")

    get_query("mistral")

