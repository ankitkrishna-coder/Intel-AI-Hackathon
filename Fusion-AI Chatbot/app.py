import os
import random
import time
from gtts import gTTS
from PIL import Image
import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from dotenv import load_dotenv
import pyperclip
from translate import Translator

# Load environment variables
load_dotenv()

# Ensure the output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Configure Streamlit app
st.set_page_config(page_title="Fusion-AI Chatbot", page_icon="🤖")

# Configure Google AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.title("Fusion-AI Chatbot 🤖")

# CSS styling for uniform button sizes
st.markdown(
    """
    <style>
    .response-button {
        width: 100%;
        height: 50px;
        font-size: 16px;
        font-weight: bold;
        color: white;
        background-color: #2a9d8f;
        border-radius: 10px;
        border: none;
        cursor: pointer;
        margin-top: 10px;
    }
    .response-button:hover {
        background-color: #21867a;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Helper functions
def write_stream(prompt, image=None, safety_settings=None, generation_config=None):
    message_placeholder = st.empty()
    message_placeholder.markdown("Thinking...")
    try:
        full_response = ""
        if image:
            chunks = st.session_state.chat_model.send_message(
                [image, prompt],
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=True,
            )
        else:
            chunks = st.session_state.chat_model.send_message(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=True,
            )
        chunks.resolve()

        for chunk in chunks:
            word_count = 0
            random_int = random.randint(5, 10)
            for word in chunk.text:
                full_response += word
                word_count += 1
                if word_count == random_int:
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "_")
                    word_count = 0
                    random_int = random.randint(5, 10)
        message_placeholder.markdown(full_response)
        return full_response

    except genai.types.generation_types.BlockedPromptException as e:
        st.error(f"Blocked Prompt: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def text_to_speech(text, lang='en'):
    supported_languages = {
        'en': 'en', 'hi': 'hi', 'bn': 'bn', 'te': 'te', 'ta': 'ta',
        'gu': 'gu', 'kn': 'kn', 'pa': 'pa', 'ml': 'ml', 'es': 'es', 'fr': 'fr',
        'de': 'de', 'it': 'it', 'pt': 'pt', 'ar': 'ar', 'ru': 'ru'
    }
    if lang not in supported_languages:
        lang = 'en'  # Fallback to English if language is unsupported

    tts = gTTS(text, lang=lang)
    temp_path = os.path.join(output_dir, "response.mp3")
    tts.save(temp_path)
    return temp_path

def save_response_to_file(response_text):
    temp_path = os.path.join(output_dir, "response.txt")
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(response_text)
    return temp_path

def translate_text(text, target_language):
    translator = Translator(to_lang=target_language)
    translation = translator.translate(text)
    return translation

# Initialize session state
if "system_instruction" not in st.session_state:
    st.session_state.system_instruction = """You are a helpful AI assistant and are talkative, proficient in both English and Vietnamese languages, and provide lots of specific details from your context. If you do not know the answer to a question, you truthfully say you do not know."""

if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel("gemini-1.5-pro-latest", system_instruction=st.session_state.system_instruction)
    st.session_state.chat_model = st.session_state.model.start_chat()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi there. Can I help you?"}]

# Default to English if no language is selected
if "selected_language" not in st.session_state:
    st.session_state.selected_language = "en"

# Sidebar configuration
with st.sidebar:
    st.header("Select Response Language")
    languages = {
        "English": "en", "Hindi": "hi", "Bengali": "bn", "Telugu": "te", "Tamil": "ta",
        "Gujarati": "gu", "Kannada": "kn", "Punjabi": "pa", "Odia": "or", "Malayalam": "ml",
        "Spanish": "es", "French": "fr", "German": "de", "Italian": "it", "Portuguese": "pt",
        "Arabic": "ar", "Russian": "ru"
    }
    selected_language = st.selectbox("Select Language for Response", list(languages.keys()))

    if languages[selected_language] != st.session_state.selected_language:
        st.session_state.selected_language = languages[selected_language]

    st.header("Choose Your Image to Upload")
    upload_image = st.file_uploader("Upload Your Image Here", accept_multiple_files=False, type=["jpg", "png", "jpeg"], key="image_uploader")

    image = Image.open(upload_image) if upload_image else None

    st.header("Adjust Model Parameters Here")
    temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=0.1, step=0.01)
    max_token = st.number_input("Maximum Output Tokens", min_value=1, max_value=4096, value=3000)
    top_p = st.number_input("Top-P sampling", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    top_k = st.number_input("Top-K sampling", min_value=0, max_value=50, value=20, step=1)
    generation_config = genai.types.GenerationConfig(max_output_tokens=max_token, temperature=temperature, top_p=top_p, top_k=top_k)

    st.header("Adjust Safety Settings Here")

    harm_categories = {
        "Dangerous Content": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        "Harassment": HarmCategory.HARM_CATEGORY_HARASSMENT,
        "Hate Speech": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "Sexually Explicit": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    }

    harm_threshold_options = {
        "Allow All": HarmBlockThreshold.BLOCK_NONE,
        "Block Low and Above": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        "Block Medium and Above": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        "Block High": HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    safety_settings = {}
    for category_name, category in harm_categories.items():
        threshold = st.selectbox(f"**{category_name}**", harm_threshold_options.keys(), index=0, key=f"{category}_threshold")
        safety_settings[category] = harm_threshold_options[threshold]

    st.header("Response Actions")

    # Copy response feature
    if st.button("Copy Response", key="copy_response"):
        response_path = os.path.join(output_dir, "response.txt")
        if os.path.exists(response_path):
            with open(response_path, 'r', encoding='utf-8') as f:
                response_text_from_file = f.read()
            pyperclip.copy(response_text_from_file)

    if os.path.exists(os.path.join(output_dir, "response.txt")):
        with open(os.path.join(output_dir, "response.txt"), 'rb') as f:
            st.download_button("Download Text", data=f, file_name="response.txt", mime="text/plain")

    if os.path.exists(os.path.join(output_dir, "response.mp3")):
        if st.button("Listen MP3", key="listen_mp3"):
            with open(os.path.join(output_dir, "response.mp3"), 'rb') as f:
                audio_bytes = f.read()
                st.audio(audio_bytes, format='audio/mp3')

        with open(os.path.join(output_dir, "response.mp3"), 'rb') as f:
            st.download_button("Download MP3", data=f, file_name="response.mp3", mime="audio/mpeg")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Hi there. Can I help you?"}]
        st.session_state.chat_model = st.session_state.model.start_chat()

# Chat interaction
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you?"):
    st.session_state.messages.append({"role": "user", "content": prompt, "image": image})
    
    with st.chat_message("user"):
        st.markdown(prompt)
        if image:
            st.image(image, width=300)

    # Get initial response
    response_text = write_stream(prompt, image, safety_settings=safety_settings, generation_config=generation_config)

    # Translate if not English
    if st.session_state.selected_language != "en":
        response_text = translate_text(response_text, st.session_state.selected_language)

    # Add response to messages and display
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # Save response to file
    save_response_to_file(response_text)

    # Generate audio file
    text_to_speech(response_text, lang=st.session_state.selected_language)