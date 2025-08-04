import os
import logging
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

pdfs_directory = 'pdfs/'
FAISS_DB_PATH = "vector_store"

def upload_pdf(file):
    try:
        os.makedirs(pdfs_directory, exist_ok=True)
        file_path = os.path.join(pdfs_directory, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        return file_path
    except Exception as e:
        logging.warning(f"PDF upload failed:{e}")
        return None

def load_pdf(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError("No content found in PDF")
        return documents
    except Exception as e:
        logging.warning(f"Error loading PDF: {e}")
        return None

def create_chunks(documents): 
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        text_chunks = text_splitter.split_documents(documents)
        
        if not text_chunks:
            raise ValueError("No text chunks created")
        
        return text_chunks
    except Exception as e:
        logging.warning(f"Error creating chunks: {e}")
        return None

def get_embedding_model():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        logging.warning(f"Error initializing embedding model: {e}")
        return None

def create_vector_store(file_path):
    try:
        documents = load_pdf(file_path)
        if documents is None:
            return None
        
        text_chunks = create_chunks(documents)
        if text_chunks is None:
            return None
        
        embeddings = get_embedding_model()
        if embeddings is None:
            return None
        
        os.makedirs(FAISS_DB_PATH, exist_ok=True)
        faiss_db = FAISS.from_documents(text_chunks, embeddings)
        faiss_db.save_local(FAISS_DB_PATH)
        return faiss_db
    except Exception as e:
        logging.warning(f"Error creating vector store: {e}")
        return None

# Only create vector store if file exists and this is run directly
if __name__ == "__main__":
    file_path = 'universal_declaration_of_human_rights.pdf'
    if os.path.exists(file_path):
        faiss_db = create_vector_store(file_path)
        if faiss_db:
            print("Vector store created successfully!")
        else:
            print("Failed to create vector store")
    else:
        print(f"PDF file not found: {file_path}")
        faiss_db = None
else:
    # When imported, try to load existing vector store
    try:
        if os.path.exists(FAISS_DB_PATH):
            embeddings = get_embedding_model()
            if embeddings:
                faiss_db = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            else:
                faiss_db = None
        else:
            faiss_db = None
    except Exception as e:
        logging.warning(f"Error loading existing vector store: {e}")
        faiss_db = None