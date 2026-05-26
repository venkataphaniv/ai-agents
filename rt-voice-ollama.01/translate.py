import speech_recognition as sr
import ollama as ol
import pyttsx3 as pt


"""
What's happening:

- adjust_for_ambient_noise: Microphones pick up fan hums and static.
  This line tells the code to listen to the silence for 0.5 seconds to
  understand the room's baseline noise, which makes the actual recognition
  much more accurate.

- recognize_google: We are using Google's Web Speech API to convert
  audio to text. It's free and generally very accurate, though it does
  require an internet connection.

"""
def listen():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Listening... (Speak now)")
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)

            # Listen for audio input
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("Processing...")

        # Recognize speech using Google's free API
        txt = recognizer.recognize_google(audio) # type: ignore
        print(f"You said: {txt}")
        return txt

    except sr.WaitTimeoutError:
        print(f"No speech detected (timeout).")
    except sr.UnknownValueError:
        print(f"Sorry, I didn't catch that.")
    except sr.RequestError:
        print(f"Speech recognition service unavailable.")
    except Exception as e:
        print(f"An error occurred in listen(): {e}")
    return None


"""
What's happening:

- ollama.chat: This is the interface to your local Llama 3 model.
  We send a list of messages (in this case, just one from the "user")
  and wait for the model to complete the pattern.

- Latency: Since Llama 3 is running locally on your device, this might
  take a second or two, depending on your GPU/CPU, but it's completely
  private. No data is sent to a cloud server for thinking.

"""
def think(txt: str):
    if not txt:
        return 'Sorry, I have nothing to think about.'

    print("Thinking...")

    try:
        # Ensure you have pulled the model via: ollama pull llama3
        res = ol.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": txt,
                }
            ],
        )

        rtxt: str = res["message"]["content"]
        print(f"Response from AI: {rtxt}")
        return str(rtxt) if txt else 'Sorry, I have no response.'

    except Exception as e:
        print(f"An error occurred in think(): {e}")
        return "Sorry, something went wrong while thinking."


"""
What's happening:

- pyttsx3.init(): This initialises the speech engine driver on your OS
  (sapi5 on Windows, nsss on Mac, espeak on Linux).

- engine.runAndWait(): This is critical. It blocks the code execution
  until the speaking is done. Without this, the program might try to
  listen while it's still speaking, causing it to hear itself!

"""
def speak(text: str):
    if not text:
        return

    try:
        engine = pt.init()

        # Optional: Change voice properties
        voices = engine.getProperty("voices")
        if voices and type(voices) is list:
            # Try changing index 0 -> 1 for alternative voice
            engine.setProperty("voice", voices[1].id)

        engine.setProperty("rate", 175)  # Speed of speech

        engine.say(text)
        engine.runAndWait()

    except Exception as e:
        print(f"An error occurred in speak(): {e}")


def main():
    print("--- Voice Assistant Started ---")
    speak("Hello, I am ready. You can start speaking.")

    while True:
        # 1. Listen
        user_input = listen()

        # Skip if nothing heard
        if not user_input:
            continue

        # 2. Check for exit keywords
        if user_input.lower().strip() in ["exit", "stop", "quit"]:
            speak("Goodbye!")
            print("Exiting...")
            break

        # 3. Think
        rt = think(user_input)

        # 4. Speak
        speak(rt)


if __name__ == "__main__":
    main()

