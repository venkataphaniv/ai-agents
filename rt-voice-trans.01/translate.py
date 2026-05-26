# Importing necessary modules required
from playsound3 import playsound
import speech_recognition as sr
from googletrans import Translator as GT
from googletrans.models import Detected, Translated
from gtts import gTTS # google text to speech api
import os, sys
import pyttsx3 as t3  # conversion of text to speech
from langdetect import detect # to know what language was used
import pycountry as ct
import asyncio as aio



# A tuple containing all the language - codes of the language will be detected
all_languages = ('afrikaans', 'af', 'albanian', 'sq', 'amharic', 'am', 'arabic', 'ar', 'armenian', 'hy', \
                'azerbaijani', 'az', 'basque', 'eu', 'belarusian', 'be','bengali', 'bn', 'bosnian', 'bs', \
                'bulgarian','bg', 'catalan', 'ca', 'cebuano','ceb', 'chichewa', 'ny', 'chinese (simplified)', \
                'zh-cn', 'chinese (traditional)', 'zh-tw', 'corsican', 'co', 'croatian', 'hr','czech', 'cs', \
                'danish', 'da', 'dutch', 'nl', 'english', 'en', 'esperanto', 'eo','estonian', 'et', \
                'filipino', 'tl', 'finnish', 'fi', 'french', 'fr', 'frisian', 'fy', 'galician', 'gl', \
                'georgian', 'ka', 'german', 'de', 'greek', 'el', 'gujarati', 'gu', 'haitian creole', 'ht', \
                'hausa', 'ha','hawaiian', 'haw', 'hebrew', 'he', 'hindi', 'hi', 'hmong', 'hmn', \
                'hungarian', 'hu', 'icelandic', 'is', 'igbo', 'ig', 'indonesian','id', 'irish', 'ga', \
                'italian','it', 'japanese', 'ja', 'javanese', 'jw', 'kannada', 'kn', 'kazakh', 'kk', \
                'khmer','km', 'korean', 'ko', 'kurdish (kurmanji)', 'ku', 'kyrgyz', 'ky', 'lao', 'lo', \
                'latin', 'la', 'latvian', 'lv', 'lithuanian', 'lt', 'luxembourgish', 'lb','macedonian', 'mk', \
                'malagasy', 'mg', 'malay','ms', 'malayalam', 'ml', 'maltese','mt', 'maori', 'mi', \
                'marathi', 'mr', 'mongolian','mn', 'myanmar (burmese)', 'my', 'nepali', 'ne', 'norwegian', 'no', \
                'odia', 'or', 'pashto', 'ps', 'persian', 'fa', 'polish', 'pl', 'portuguese', 'pt', 'punjabi','pa', \
                'romanian', 'ro', 'russian', 'ru', 'samoan', 'sm', 'scots gaelic', 'gd', 'serbian', 'sr', \
                'sesotho', 'st','shona', 'sn', 'sindhi', 'sd', 'sinhala', 'si', 'slovak', 'sk', 'slovenian', 'sl', \
                'somali', 'so', 'spanish', 'es', 'sundanese','su', 'swahili', 'sw', 'swedish', 'sv', 'tajik', 'tg', \
                'tamil', 'ta', 'telugu', 'te', 'thai', 'th', 'turkish', 'tr', 'ukrainian', 'uk', 'urdu', 'ur', \
                'uyghur', 'ug', 'uzbek', 'uz', 'vietnamese', 'vi', 'welsh', 'cy', 'xhosa', 'xh','yiddish', 'yi', \
                'yoruba', 'yo', 'zulu', 'zu')

all_lang_dict = {'afrikaans': 'af', 'albanian': 'sq', 'amharic': 'am', 'arabic': 'ar', 'armenian': 'hy', \
                'azerbaijani': 'az', 'basque': 'eu', 'belarusian': 'be', 'bengali': 'bn', 'bosnian': 'bs', \
                'bulgarian': 'bg', 'catalan': 'ca', 'cebuano':'ceb', 'chichewa': 'ny', 'chinese (simplified)': \
                'zh-cn', 'chinese (traditional)': 'zh-tw', 'corsican': 'co', 'croatian': 'hr', 'czech': 'cs', \
                'danish': 'da', 'dutch': 'nl', 'english': 'en', 'esperanto': 'eo', 'estonian': 'et', \
                'filipino': 'tl', 'finnish': 'fi', 'french': 'fr', 'frisian': 'fy', 'galician': 'gl', \
                'georgian': 'ka', 'german': 'de', 'greek': 'el', 'gujarati': 'gu', 'haitian creole': 'ht', \
                'hausa': 'ha', 'hawaiian': 'haw', 'hebrew': 'he', 'hindi': 'hi', 'hmong': 'hmn', \
                'hungarian': 'hu', 'icelandic': 'is', 'igbo': 'ig', 'indonesian': 'id', 'irish': 'ga', \
                'italian': 'it', 'japanese': 'ja', 'javanese': 'jw', 'kannada': 'kn', 'kazakh': 'kk', \
                'khmer': 'km', 'korean': 'ko', 'kurdish (kurmanji)': 'ku', 'kyrgyz': 'ky', 'lao': 'lo', \
                'latin': 'la', 'latvian': 'lv', 'lithuanian': 'lt', 'luxembourgish': 'lb','macedonian': 'mk', \
                'malagasy': 'mg', 'malay':'ms', 'malayalam': 'ml', 'maltese':'mt', 'maori': 'mi', \
                'marathi': 'mr', 'mongolian':'mn', 'myanmar (burmese)': 'my', 'nepali': 'ne', 'norwegian': 'no', \
                'odia': 'or', 'pashto': 'ps', 'persian': 'fa', 'polish': 'pl', 'portuguese': 'pt', 'punjabi': 'pa', \
                'romanian': 'ro', 'russian': 'ru', 'samoan': 'sm', 'scots gaelic': 'gd', 'serbian': 'sr', \
                'sesotho': 'st','shona': 'sn', 'sindhi': 'sd', 'sinhala': 'si', 'slovak': 'sk', 'slovenian': 'sl', \
                'somali': 'so', 'spanish': 'es', 'sundanese':'su', 'swahili': 'sw', 'swedish': 'sv', 'tajik': 'tg', \
                'tamil': 'ta', 'telugu': 'te', 'thai': 'th', 'turkish': 'tr', 'ukrainian': 'uk', 'urdu': 'ur', \
                'uyghur': 'ug', 'uzbek': 'uz', 'vietnamese': 'vi', 'welsh': 'cy', 'xhosa': 'xh','yiddish': 'yi', \
                'yoruba': 'yo', 'zulu': 'zu'}



def speak(eng: t3.Engine, audio):
    eng.say(audio)
    eng.runAndWait()


# get the language name from language code
def get_lang_name(lc: str) -> str:
    lang = ct.languages.get(alpha_2 = lc)
    return lang.name if lang is not None else 'Unknown'


# get destination language / language to convert to from user
def get_dest_lang():
    print("Enter the language in which you	want to convert : Ex. Hindi , English , Spanish, etc.\n")

    # Input destination language in which the user wants to translate
    dtl = take_command()
    while dtl == 'NA':
        print("Please say the language again")
        dtl = take_command()
    dtl = dtl.lower()
    return dtl, all_lang_dict.get(dtl, '')


# Capture Voice
# takes command through microphone
def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("listening.....")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing.....")
        query = r.recognize_google(audio, language='en-in') # type: ignore
        print(f"The User said {query}\n")
    except Exception as e:
        print("say that again please.....")
        return "NA"
    return query


# get SAPI 5 translation engine
def get_engine():
    te: t3.Engine = t3.init('sapi5')

    if type(te) is t3.Engine:
        voices = te.getProperty('voices')
        if type(voices) is list and len(voices) > 1:
            print(f"Available voices: {[voice.name for voice in voices]}")
            te.setProperty('voice', voices[1].id)
        return te
    return None


def get_input():
    # Input from user and make input to lowercase
    ip = take_command()

    while (ip == "NA"):
        ip = take_command()
    return ip.lower()


def get_text(te: t3.Engine):
    print ("Welcome to the translator! ")
    speak (te, "Welcome to the translator! ")
    print ("Say the sentence you want to translate once you see the word 'listening'")


def translate(tl: str, query: str):
    # invoking Google Translator
    tr = GT()

    # Translating from src to dest language using translate method
    # capture into text to translate
    tt: Translated = aio.run(tr.translate(query, dest=tl))
    return tt.text


if __name__ == "__main__":
    # get SAPI 5 translation engine
    te: t3.Engine | None = get_engine()

    if te is not None:
        # get text to translate
        get_text(te)

        # take input from user
        ip = get_input()

        il = detect(ip)

        # detect the language of the input
        print (f"The user's sentence is in {get_lang_name(il)}")

        # get the language code for destination language
        dl, dlc = get_dest_lang()
        while (dl not in all_languages):
            print("Language in which you are trying	to convert is currently not available, please input some other language\n")
            dl, dlc = get_dest_lang()

        print (f"Translating to {dl} language, with code {dlc}\n")

        # Calling the translate function
        ttxt = translate(dlc, ip)

        # Using Google-Text-to-Speech ie, gTTS() method to speak the translated text into
        # the destination language which is stored in to_lang.
        # Also, we have given 3rd argument as False because by default it speaks very slowly
        spk = gTTS(text=ttxt, lang=dlc, slow=False)

        print(spk.text)

        # Using save() method to save the translated speech in a file
        spk.save("audio.mp3")

        # Using OS module to run the translated voice.
        playsound('audio.mp3')
        os.remove('audio.mp3')

        # Printing Output
        print(ttxt)

