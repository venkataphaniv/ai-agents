# LinguaSync: Real-Time Voice Translator

<!-- <div align="center">
 <a href="#"><img alt="language" src="https://user-images.githubusercontent.com/132539454/278971782-9453805e-e2e6-4d99-b1de-cf8fcd3e7105.svg"></a>
</div> -->

Real-Time Voice Translator is a machine learning project that aims to provide a seamless and natural experience of cross-lingual communication. It uses deep neural networks to translate voice from one language to another in real time while preserving the tone and emotion of the speaker. It is a desktop application that supports Windows, Linux, and Mac operating systems.

The application is easy to use: simply select the languages you want to translate between and start speaking. The application will listen to your voice and provide instant translations in real-time. You can also use the application to translate conversations between two or more people.

## Dependencies

    <=Python3.11, gTTS, pyaudio, playsound==1.2.2, deep-translator, SpeechRecognition, google-transliteration-api, cx-Freeze

## Getting started

1. Clone this project and create virtualenv (recommended) and activate virtualenv

        # Create virtualenv
        python -m venv env

        # Linux/MacOS
        source env/bin/activate

        # Windows
        env\Scripts\activate

2. Install require dependencies

        pip install --upgrade wheel

        pip install -r requirements.txt

3. Run code and speech (have fun).

        python main.py

### Program Flow

![Block Diagram of Voice Translator](flow.png)

### Install Windows/Linux/Mac Application

Code: <https://github.com/SamirPaulb/real-time-voice-translator/releases/latest>

I am using cx_Freeze <https://github.com/marcelotduarte/cx_Freeze/tree/main> to build executable file of this app.
The build settings can be changed by modifying the setup.py file.

### Build installer containing all the files

- Windows: ```python setup.py bdist_msi```
- Linux: ```python setup.py bdist_rpm```
- Mac: ```python setup.py bdist_mac```

### GUI

![App GUI](gui.png)
