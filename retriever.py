from vector_store import create_vector_store

def create_retriever(vector_store, search_type="mmr", k=1):
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )
    return retriever

def batch_retrieve(retriever, queries):
    results = retriever.batch(queries)
    return results