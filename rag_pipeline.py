import logging
from langchain_groq import ChatGroq
from vector_db import faiss_db
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv; load_dotenv()

try:
    import os
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    llm_model = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        api_key=api_key
    )
except Exception as e:
    logging.warning(f"Error initializing ChatGroq model: {e}")
    llm_model = None

def retrieve_docs(query):
    try:
        return faiss_db.similarity_search(query)
    except Exception as e:
        logging.warning(f"Error retrieving documents: {e}")
        return []

def get_context(documents):
    try:
        context = "\n\n".join([doc.page_content for doc in documents])
        return context
    except Exception as e:
        logging.warning(f"Error getting context: {e}")
        return ""

custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context
Question: {question} 
Context: {context} 
Answer:
"""

def answer_query(documents, model, query):
    try:
        if model is None:
            return "Error: Language model not initialized"
        
        context = get_context(documents)
        if not context:
            return "No relevant context found for your question."
        
        prompt = ChatPromptTemplate.from_template(custom_prompt_template)
        chain = prompt | model
        response = chain.invoke({"question": query, "context": context})
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        logging.warning(f"Error generating answer: {e}")
        return "Sorry, I encountered an error while processing your question."
