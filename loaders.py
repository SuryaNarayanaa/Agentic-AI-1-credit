import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import UnstructuredExcelLoader

loader = UnstructuredExcelLoader("./Daily_Income_Expenditure_2025.xlsx", mode="elements")
docs = loader.load()

# docs
print((docs))


page_url = "https://python.langchain.com/docs/how_to/chatbots_memory/"

loader = WebBaseLoader(web_paths=[page_url])
docs = []
for doc in loader.lazy_load():
    docs.append(doc)

assert len(docs) == 1
doc = docs[0]
print(doc.page_content[:1000])