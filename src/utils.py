from groq import Groq
import os
import base64
import streamlit as st
from google.cloud import texttospeech
import json
from gtts import gTTS

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))

GROQ_API_KEY = config_data["GROQ_API_KEY"]

# Save the API key to environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

client = Groq()

# Generate answers using Groq's chatbot model
def get_answer(messages):
    system_message = [{"role": "system", "content": "You are a helpful AI chatbot that answers questions asked by the User."}]
    messages = system_message + messages
    response = client.chat.completions.create(  
        model="llama-3.1-8b-instant",  # Update to Groq's appropriate model ID
        messages=messages
    )
    return response.choices[0].message.content

# Speech-to-text using Groq's model
def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript_response = client.audio.transcriptions.create(  # Example method name, adjust based on actual Groq API
            model="distil-whisper-large-v3-en",
            prompt="Specify context or spelling",
            response_format="text",
            language="en",
            temperature=0.0,
            file=audio_file
        )
    
    # Ensure transcript_response is processed correctly
    if isinstance(transcript_response, str):
        return transcript_response
    else:
        return transcript_response.get('text', '')

# Text-to-speech using Googles gtts model

def text_to_speech(input_text):
    # Create a gTTS object
    tts = gTTS(text=input_text, lang='en', slow=False)

    webm_file_path = "temp_audio_play.mp3"
    
    # Save the audio file
    tts.save(webm_file_path)

    return webm_file_path

# Autoplay audio in the Streamlit app
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)
