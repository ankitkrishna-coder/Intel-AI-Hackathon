import os
import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from gtts import gTTS
import pyperclip
import urllib.parse
from translate import Translator

# Directory for saving outputs
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Multilingual supported languages
supported_languages = {
    'English': 'en',
    'Hindi': 'hi',
    'Bengali': 'bn',
    'Telugu': 'te',
    'Tamil': 'ta',
    'Gujarati': 'gu',
    'Kannada': 'kn',
    'Punjabi': 'pa',
    'Malayalam': 'ml',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Arabic': 'ar',
    'Russian': 'ru'
}

# Function to connect to either MySQL or PostgreSQL databases
def connectDatabase(db_type, username, port, host, password, database):
    password = urllib.parse.quote_plus(password)
    if db_type == "MySQL":
        uri = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}"
    elif db_type == "PostgreSQL":
        uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
    else:
        st.error("Unsupported database type")
        return
    st.session_state.db = SQLDatabase.from_uri(uri)
    st.sidebar.success("Database connected!")

# Function to run a SQL query
def runQuery(query):
    return st.session_state.db.run(query) if st.session_state.db else "Please connect to the database first."

# Function to retrieve the database schema
def getDatabaseSchema():
    return st.session_state.db.get_table_info() if st.session_state.db else "Please connect to the database."

# LLM model
llm = ChatOllama(model="llama3")

# Function to generate SQL query from user's natural language question
def getQueryFromLLM(question):
    template = """Below is the schema of the connected database. Read the schema carefully, paying attention to the table and column names. Ensure that table or column names are used exactly as they appear in the schema. Answer the user's question in the form of an SQL query.

    {schema}

    Please provide only the SQL query and nothing else.

    question: {question}
    SQL query:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    response = chain.invoke({
        "question": question,
        "schema": getDatabaseSchema()
    })
    return response.content

# Function to generate a response for the query result
def getResponseForQueryResult(question, query, result):
    template = """Below is the schema of the connected database. Based on the schema and the query result, write a natural language response to the user's question.

    {schema}

    Examples:
    question: How many albums are in the database?
    SQL query: SELECT COUNT(*) FROM album;
    Result: [(34,)]
    Response: There are 34 albums in the database.

    Now it's your turn to write a response based on the result:
    question: {question}
    SQL query: {query}
    Result: {result}
    Response:"""
    
    prompt2 = ChatPromptTemplate.from_template(template)
    chain2 = prompt2 | llm

    response = chain2.invoke({
        "question": question,
        "schema": getDatabaseSchema(),
        "query": query,
        "result": result
    })

    return response.content

# Function to save response to a text file
def save_response_to_file(response_text):
    temp_path = os.path.join(output_dir, "response.txt")
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(response_text)
    return temp_path

# Function to generate and save text-to-speech audio
def text_to_speech(text, lang='en'):
    tts = gTTS(text, lang=lang)
    temp_path = os.path.join(output_dir, "response.mp3")
    tts.save(temp_path)
    return temp_path

# Function to translate the response
def translate_text(text, target_language):
    translator = Translator(to_lang=target_language)
    translation = translator.translate(text)
    return translation

# Streamlit app configuration
st.set_page_config(
    page_icon="🤖",
    page_title="Fusion-AI Database Chatbot",
    layout="centered"
)

st.title("Fusion-AI Database Chatbot 🤖")

# Sidebar for database connection inputs
with st.sidebar:
    # Language selection for responses
    st.title('Response Language')
    selected_language = st.selectbox("Select Language", options=list(supported_languages.keys()), key="language")
    
    st.title('Connect to a Database')
    db_type = st.selectbox("Database Type", options=["MySQL", "PostgreSQL"], key="db_type")
    host = st.text_input("Host", value="localhost", key="host")
    port = st.text_input("Port", value="3306" if db_type == "MySQL" else "5432", key="port")
    username = st.text_input("Username", value="", key="username")
    password = st.text_input("Password", value="", type="password", key="password")
    database = st.text_input("Database", value="", key="database")
    connectBtn = st.button("Connect")

    if connectBtn:
        connectDatabase(db_type, username, port, host, password, database)
 
    st.title('Response Actions')

    # Download and Copy Features
    if st.button("Copy Response to Clipboard"):
        if os.path.exists(os.path.join(output_dir, "response.txt")):
            with open(os.path.join(output_dir, "response.txt"), 'r', encoding='utf-8') as f:
                response_text_from_file = f.read()
            pyperclip.copy(response_text_from_file)
            st.success("Response copied to clipboard")

    if os.path.exists(os.path.join(output_dir, "response.txt")):
        with open(os.path.join(output_dir, "response.txt"), 'rb') as f:
            st.download_button("Download Response (Text)", data=f, file_name="response.txt", mime="text/plain")

    if os.path.exists(os.path.join(output_dir, "response.mp3")):
        if st.button("Listen to Response (Audio)"):
            with open(os.path.join(output_dir, "response.mp3"), 'rb') as f:
                audio_bytes = f.read()
                st.audio(audio_bytes, format='audio/mp3')

        with open(os.path.join(output_dir, "response.mp3"), 'rb') as f:
            st.download_button("Download Response (Audio)", data=f, file_name="response.mp3", mime="audio/mpeg")

    if st.button("Clear Chat History"):
        st.session_state.chat = []

# Main chat interface
if "chat" not in st.session_state:
    st.session_state.chat = []

question = st.chat_input("Chat with your database")

if question:
    if "db" not in st.session_state:
        st.error('Please connect to the database first.')
    else:
        st.session_state.chat.append({"role": "user", "content": question})

        # Generate SQL query from LLM
        query = getQueryFromLLM(question)

        # Execute the query
        result = runQuery(query)

        # Generate a response based on the query result
        response = getResponseForQueryResult(question, query, result)

        # Translate the response if a language other than English is selected
        target_language_code = supported_languages.get(selected_language, 'en')
        if target_language_code != 'en':
            response = translate_text(response, target_language_code)
        
        st.session_state.chat.append({"role": "assistant", "content": response})

        # Save the response and generate audio
        save_response_to_file(response)
        text_to_speech(response, lang=target_language_code)

# Display chat messages
for chat in st.session_state.chat:
    st.chat_message(chat['role']).markdown(chat['content'])
