import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
from datetime import datetime
import glob
import re

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state for chat history and selected chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_chat" not in st.session_state:
    st.session_state.selected_chat = None

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def process_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    return response["output_text"]

def clean_filename(text):
    # Remove any special characters and limit length
    clean_text = re.sub(r'[^\w\s-]', '', text)
    clean_text = re.sub(r'\s+', '_', clean_text.strip())
    return clean_text[:50]  # Limit to first 50 characters

def save_chat_history(chat_history):
    if not os.path.exists('chat_histories'):
        os.makedirs('chat_histories')
    
    # Get the first question from chat history
    first_question = "chat"
    for message in chat_history:
        if message["role"] == "user":
            first_question = clean_filename(message["content"])
            break
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_histories/{first_question}_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=4)
    
    return filename

def load_chat_history(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_saved_chats():
    if not os.path.exists('chat_histories'):
        return []
    return sorted(glob.glob('chat_histories/*.json'), reverse=True)

def format_filename_to_display(filename):
    try:
        # Extract the question part and timestamp
        base_name = os.path.basename(filename)
        question_part = base_name.rsplit('_', 1)[0]
        timestamp_str = base_name.split('_')[-1].replace('.json', '')
        
        # Format the timestamp
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        formatted_date = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Replace underscores with spaces in question part
        question_part = question_part.replace('_', ' ')
        
        return f"{question_part} ({formatted_date})"
    except:
        return filename

def main():
    st.set_page_config("Fusion-AI Multi-PDF Chatbot 🤖")
    st.title("Fusion-AI Multi-PDF Chatbot 🤖")

    # Sidebar for PDF upload and controls
    with st.sidebar:
        
        # PDF Upload Section
        st.header("📄 Upload Documents")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
        
        st.divider()
        
        # Chat Controls Section
        st.header("⚙️ Chat Controls")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("💾 Save Chat"):
                if st.session_state.chat_history:
                    filename = save_chat_history(st.session_state.chat_history)
                    # st.success("Chat saved!")
                else:
                    st.warning("No chat history to save!")
        
        st.divider()
        
        # Saved Chats Section
        st.header("💬 Saved Chats")
        saved_chats = get_saved_chats()
        
        if saved_chats:
            selected_chat = st.selectbox(
                "Select a saved chat:",
                saved_chats,
                format_func=format_filename_to_display
            )
            
            if selected_chat:
                chat_data = load_chat_history(selected_chat)
                for msg in chat_data[:3]:
                    st.markdown(f"**{msg['role']}**: {msg['content'][:50]}...")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Load Chat"):
                        st.session_state.chat_history = chat_data
                        st.rerun()
                with col2:
                    if st.button("Delete Chat"):
                        os.remove(selected_chat)
                        st.success("Chat deleted!")
                        st.rerun()
        else:
            st.info("No saved chats found")

    # Main chat area
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if user_question := st.chat_input("Ask a question about your documents"):
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)
        
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_question
        })

        # Display thinking message
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.text("Thinking...")
            
            try:
                # Generate response
                response = process_user_input(user_question)
                # Replace thinking message with actual response
                thinking_placeholder.empty()
                st.write(response)
                
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response
                })
                
            except FileNotFoundError:
                thinking_placeholder.empty()
                st.error("Please upload and process PDF documents first!")
            except Exception as e:
                thinking_placeholder.empty()
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()