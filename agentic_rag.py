from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_store import create_vector_store, add_documents_to_store
from retriever import create_retriever
from agents import get_gemini
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
import re

# ---------------------------
# Setup
# ---------------------------
vector_store = create_vector_store()
response_model = get_gemini()

# Existing retriever tool
name = "retriever_policies"
description = "Search for the provided content and answer questions about it."
retriever = create_retriever(vector_store, search_type="mmr", k=1)
retriever_tool = create_retriever_tool(retriever, name, description)

# ---------------------------
# Tool: Load URLs into vector store
# ---------------------------
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

@tool
def load_url_tool(url: str):
    """Load and clean content from a given URL and add to vector store."""
    loader = WebBaseLoader(url)
    documents = loader.load()

    html2text = Html2TextTransformer()
    documents = html2text.transform_documents(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    add_documents_to_store(vector_store, splits)

    return f"Content from {url} was extracted and added to the vector store."

def detect_urls(text: str):
    """Detect URLs in the text."""
    url_pattern = r'https?://[^\s]+'
    return re.findall(url_pattern, text)
def generate_query_or_respond(state: MessagesState):
    """Handle user queries: if URLs provided, load them and respond directly."""
    user_message = state["messages"][-1].content
    urls = detect_urls(user_message)

    print("Detected URLs:", urls)

    if urls:
        # Load all URLs into vector store
        for url in urls:
            load_url_tool.invoke({"url": url})

        # After loading, generate answer directly
        response = response_model.bind_tools([retriever_tool]).invoke(
            state["messages"]
        )
        return {"messages": [response], "next": END}  # Force finish

    else:
        # No URLs → normal flow (tools_condition will decide)
        response = response_model.bind_tools([retriever_tool]).invoke(
            state["messages"]
        )
        return {"messages": [response]}


def rewrite_question(state: MessagesState):
    """Rewrite the question for better retrieval."""
    # Placeholder implementation
    return {"messages": state["messages"]}


def generate_answer(state: MessagesState):
    """Generate the final answer."""
    # Placeholder implementation
    return {"messages": state["messages"]}


def grade_documents(state: MessagesState):
    """Grade the retrieved documents."""
    # Placeholder implementation - assume documents are good
    return "generate_answer"

# ---------------------------
# Workflow Graph
# ---------------------------
workflow = StateGraph(MessagesState)

workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate_answer": "generate_answer",
        "rewrite_question": "rewrite_question",
    },
)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")
graph = workflow.compile()
input_state = {
    "messages": [
        {
            "role": "user",
            "content": "What are the leave policies? Check this URL: https://en.wikipedia.org/wiki/Leave_of_absence",
        }
    ]
}

result = graph.invoke(input_state)
print(result)
