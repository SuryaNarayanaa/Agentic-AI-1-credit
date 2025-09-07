import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embeddings():
    google_api_key = os.getenv("GEMINI_API_KEY")
    if not google_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=google_api_key
    )
    return embeddings