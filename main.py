import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from vector_db import (
    upload_pdf,
    load_pdf,
    create_chunks,
    get_embedding_model,
    create_vector_store
)
from rag_pipeline import retrieve_docs, get_context, answer_query

load_dotenv()
load_dotenv('.env')  # Try explicit path
load_dotenv(os.path.join(os.getcwd(), '.env'))  # Try absolute path

# Manual fallback - read .env file directly if load_dotenv fails
if not os.getenv("GROQ_API_KEY"):
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('GROQ_API_KEY='):
                    key_value = line.strip().split('=', 1)
                    if len(key_value) == 2:
                        os.environ['GROQ_API_KEY'] = key_value[1]
                        break
    except FileNotFoundError:
        pass

# Check if API key is loaded
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("❌ GROQ_API_KEY not found in environment variables. Please check your .env file.")
    st.info("Make sure your .env file contains: GROQ_API_KEY=your_api_key_here")
    st.stop()

custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context
Question: {question} 
Context: {context} 
Answer:
"""

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_DB_PATH = "vector_store"
pdfs_directory = 'pdfs/'

# Ensure directories exist
os.makedirs(pdfs_directory, exist_ok=True)
os.makedirs(FAISS_DB_PATH, exist_ok=True)

try:
    llm_model = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        api_key=api_key
    )
    st.success("✅ ChatGroq model initialized successfully!")
except Exception as e:
    st.error(f"❌ Failed to initialize ChatGroq model: {e}")
    st.info("Please check your GROQ_API_KEY in the .env file")
    st.stop()


# Streamlit UI
st.title("🤖 AI Legal Assistant")
st.write("Upload a PDF document and ask questions about its content!")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

user_query = st.text_area("Enter your prompt: ", height=150, placeholder="Ask Anything!")

ask_question = st.button("Ask AI Lawyer")

if ask_question:
    if uploaded_file and user_query:
        with st.spinner("Processing your document..."):
            # Upload and process PDF
            file_path = upload_pdf(uploaded_file)
            if file_path is None:
                st.stop()
            
            documents = load_pdf(file_path)
            if documents is None:
                st.stop()
            
            text_chunks = create_chunks(documents)
            if text_chunks is None:
                st.stop()
            
            faiss_db = create_vector_store(FAISS_DB_PATH, text_chunks)
            if faiss_db is None:
                st.stop()
            
            # Retrieve relevant documents and generate answer
            retrieved_docs = retrieve_docs(faiss_db, user_query)
            response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)
            
            # Display results
            st.chat_message("user").write(user_query)
            st.chat_message("AI Lawyer").write(response)
    else:
        st.error("Please upload a valid PDF file and enter a question!")
