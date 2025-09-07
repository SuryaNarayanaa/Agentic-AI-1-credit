import os
from dotenv import load_dotenv
import retriever
from splitter import split_document
from vector_store import create_vector_store, add_documents_to_store, similarity_search
from retriever import create_retriever, batch_retrieve
from langchain.tools.retriever import create_retriever_tool

# Load environment variables
load_dotenv()

def main():
    # Sample content to split into documents
    sample_content = """
    Microsoft Office is a suite of productivity applications including Word, Excel, PowerPoint, and Outlook.
    It can be installed on Windows and Mac computers. The installation requires a valid license key.
    
    Leave options in our company include annual leave, sick leave, maternity leave, and paternity leave.
    Employees can apply for leave through the HR portal.
    
    The CEO of our company is John Smith, who has been leading the organization for 10 years.
    He holds a degree in Business Administration and has extensive experience in the tech industry.
    """

    # Split the content into documents
    all_splits = split_document(sample_content, chunk_size=200, chunk_overlap=50)

    print(f"Number of document chunks: {len(all_splits)}")

    # Create vector store
    vector_store = create_vector_store()

    # Add documents to vector store
    ids = add_documents_to_store(vector_store, all_splits)
    print(f"Added {len(ids)} documents to vector store")

    # Perform similarity search
    query = "can i install microsoft office"
    results = similarity_search(vector_store, query, k=1)
    print(f"Similarity search results for '{query}':")
    print(results[0].page_content if results else "No results")

    # Create retriever
    retriever = create_retriever(vector_store, search_type="mmr", k=1)
    
    retriever_tool = create_retriever_tool(
        retriever,
        "retriever_policies",
        "Search for company policies for onboarding and HR related queries",
    )
    # Perform batch retrieval
    queries = [
        "what is my leave options",
        "Who is the CEO of the company",
    ]
    retriever_tool.invoke({"query": "business policies"})
    # batch_results = batch_retrieve(retriever, queries)
    # print("\nBatch retrieval results:")
    # for i, result in enumerate(batch_results):
    #     print(f"Query {i+1}: {queries[i]}")
    #     if result:
    #         print(f"Result: {result[0].page_content}")
    #     else:
    #         print("No results")
    #     print()
    print(retriever_tool)
if __name__ == "__main__":
    main()