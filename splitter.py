from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def split_document(content: str, chunk_size: int = 500, chunk_overlap: int = 50, url: str = None, file_name: str = None) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(content)
    documents = []
    for i, chunk in enumerate(chunks):
        metadata = {"chunk_index": i}
        if url:
            metadata["source"] = url
        if file_name:
            metadata["file_name"] = file_name
        doc = Document(page_content=chunk, metadata=metadata)
        documents.append(doc)
    return documents

