from langchain_core.vectorstores import InMemoryVectorStore
from embeddings import get_embeddings

def create_vector_store():
    embeddings = get_embeddings()
    vector_store = InMemoryVectorStore(embeddings)
    return vector_store

def add_documents_to_store(vector_store, documents):
    ids = vector_store.add_documents(documents=documents)
    return ids

def similarity_search(vector_store, query, k=1):
    results = vector_store.similarity_search(query, k=k)
    return results