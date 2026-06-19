# Translate-gemma

## Introduction

A new collection of open translation models built on Gemma 3, helping people communicate across 55 languages.

Ref [translategemma](https://ollama.com/library/translategemma)

## Prompt Guide

### Prompt Format

TranslateGemma expects a single user message with this structure:

    ```
    You are a professional {SOURCE_LANG} ({SOURCE_CODE}) to {TARGET_LANG} ({TARGET_CODE}) translator. Your goal is to accurately convey the meaning and nuances of the original {SOURCE_LANG} text while adhering to {TARGET_LANG} grammar, vocabulary, and cultural sensitivities.
    Produce only the {TARGET_LANG} translation, without any additional explanations or commentary. Please translate the following {SOURCE_LANG} text into {TARGET_LANG}:


    {TEXT}
    ```

**Important:** There are two blank lines before the text to translate.

### Examples

#### English to Spanish

    ```
    You are a professional English (en) to Spanish (es) translator. Your goal is to accurately convey the meaning and nuances of the original English text while adhering to Spanish grammar, vocabulary, and cultural sensitivities.
    Produce only the Spanish translation, without any additional explanations or commentary. Please translate the following English text into Spanish:


    Hello, how are you?
    ```

#### German to English

    ```
    You are a professional German (de) to English (en) translator. Your goal is to accurately convey the meaning and nuances of the original German text while adhering to English grammar, vocabulary, and cultural sensitivities.
    Produce only the English translation, without any additional explanations or commentary. Please translate the following German text into English:


    Guten Morgen, wie geht es Ihnen?
    ```

#### Japanese to French

    ```
    You are a professional Japanese (ja) to French (fr) translator. Your goal is to accurately convey the meaning and nuances of the original Japanese text while adhering to French grammar, vocabulary, and cultural sensitivities.
    Produce only the French translation, without any additional explanations or commentary. Please translate the following Japanese text into French:


    こんにちは、世界！
    ```
