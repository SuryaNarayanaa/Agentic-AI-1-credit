from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain.chat_models import init_chat_model
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_store import create_vector_store, add_documents_to_store
from retriever import create_retriever
from agents import get_gemini
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import MessagesState
import re

vector_store = create_vector_store()
response_model = get_gemini()

name = "retriever_policies"
description = "Search for company policies for onboarding and HR related queries"
retriever = create_retriever(vector_store, search_type="mmr", k=1)
retriever_tool = create_retriever_tool(retriever, name, description)

def generate_query_or_respond(state: MessagesState):
    """Decide whether to use the retriever tool or respond directly."""
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}

def grade_documents(state: MessagesState):
    """Grade the retrieved documents for relevance."""
    # Placeholder logic: assume documents are always relevant for simplicity
    # In a real implementation, check relevance based on state
    return "generate_answer"

def rewrite_question(state: MessagesState):
    """Rewrite the question if documents are not relevant."""
    # Placeholder: rewrite the last user message
    last_message = state["messages"][-1].content
    rewritten = f"Rewritten: {last_message}"
    return {"messages": [{"role": "user", "content": rewritten}]}

def generate_answer(state: MessagesState):
    """Generate the final answer based on retrieved documents."""
    # Placeholder: use the model to generate answer
    response = response_model.invoke(state["messages"])
    return {"messages": [response]}

# Build the workflow
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

# Example usage
input_state = {
    "messages": [
        {
            "role": "user",
            "content": "What are the leave policies?",
        }
    ]
}

result = graph.invoke(input_state)
print(result)
