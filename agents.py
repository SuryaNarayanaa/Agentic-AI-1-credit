import os
import uuid
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from web_aware import scrape_wikipedia_page
from splitter import split_document
from embeddings import get_embeddings
from vector_store import create_vector_store, add_documents_to_store
from retriever import create_retriever

# ------------------------- 
# Retriever Tool Setup
# -------------------------
def create_retriever_tool_from_vector_store(vector_store, name="retriever_policies", description="Search for company policies for onboarding and HR related queries"):
    retriever = create_retriever(vector_store, search_type="mmr", k=1)
    retriever_tool = create_retriever_tool(
        retriever,
        name,
        description,
    )
    return retriever_tool

# ------------------------- 
# Generate Query or Respond Function
# -------------------------
def generate_query_or_respond(state: MessagesState, retriever_tool):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response_model = get_gemini()
    response = (
        response_model
        .bind_tools([retriever_tool]).invoke(state["messages"])
    )
    return {"messages": [response]}

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# -------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# -------------------------
# Gemini LLM Setup
# -------------------------
def get_gemini():
    if not GOOGLE_API_KEY:
        raise ValueError("Missing GEMINI_API_KEY in environment variables.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        top_k=None,
        top_p=None,
        safety_settings=None,
        api_key=GOOGLE_API_KEY,
    )
    return llm

# -------------------------
# Prompt Types
# -------------------------
def get_prompt_types():
    return [
        "tot",              # Tree of Thoughts
        "cot",              # Chain of Thoughts
        "ReAct",            # Reason + Act
        "Reflexion",        # Reflect on mistakes
        "Self-Consistency", # Multiple reasoning paths
        "Debate",           # Agents argue
        "Plan and Solve",   # Plan first, then execute
        "PaR"               # Program Aided Reasoning
    ]

def get_prompt_description(prompt_type: str) -> str:
    descriptions = {
        "tot": "Tree of Thoughts",
        "cot": "Chain of Thoughts",
        "ReAct": "Reason + Act (alternate reasoning and actions)",
        "Reflexion": "Reflect on mistakes and improve iteratively",
        "Self-Consistency": "Generate 3 reasoning paths",
        "Debate": "Multiple agents argue",
        "Plan and Solve": "Create plan and execute",
        "PaR": "Program Aided Reasoning (write and execute code)"
    }
    return descriptions.get(prompt_type, "Unknown Prompt Type")

def get_prompt_template(prompt_type: str) -> str:
    templates = {
        "tot": "You are using the Tree of Thoughts approach. Think step by step and explore multiple reasoning paths.",
        "cot": "You are using the Chain of Thoughts approach. Provide a detailed reasoning process before answering.",
        "ReAct": "You are using the ReAct approach. Alternate between reasoning and taking actions to gather information.",
        "Reflexion": "You are using the Reflexion approach. Reflect on your previous answers and improve iteratively.",
        "Self-Consistency": "You are using the Self-Consistency approach. Generate multiple reasoning paths and choose the most consistent answer.",
        "Debate": "You are using the Debate approach. Argue different perspectives before reaching a conclusion.",
        "Plan and Solve": "You are using the Plan and Solve approach. Create a plan first, then execute it step by step.",
        "PaR": "You are using the Program Aided Reasoning approach. Write and execute code to help solve the problem."
    }
    return templates.get(prompt_type, "Default prompt template.")

# -------------------------
# Agent State
# -------------------------
class AgentState(TypedDict, total=False):
    query: str
    prompted_query: str
    response: str


# -------------------------
# Graph Building
# -------------------------
def build_graph(prompt_type: str, url: str = None, file_content: str = None) -> StateGraph:

    def process_input(state: AgentState) -> AgentState:
        query = state["query"]
        prompt = get_prompt_template(prompt_type)

        print(f"\n[Prompt Type] {prompt_type}")
        print(f"[Prompt Template]\n{prompt}\n")

        if file_content:
            chunks = split_document(file_content)
            # Take only first 4 documents
            docs = chunks[:4]
            if docs:
                # Use new modular vector store system
                vector_store = create_vector_store()
                ids = add_documents_to_store(vector_store, docs)
                retriever = create_retriever(vector_store, search_type="mmr", k=2)
                print(f"[File Content] Using {len(docs)} chunks from uploaded file for context.")
                print(f"[File Content] Added {len(ids)} documents to vector store")
                
                # Retrieve relevant docs using retriever
                relevant_docs = retriever.invoke(query)
                print(f"[File Content] Retrieved {len(relevant_docs)} relevant chunks for context.")
                context = "\n".join([doc.page_content for doc in relevant_docs])
                prompt = f"""
                    You are an assistant with access to uploaded file content. 
                    Use the following relevant context to answer the user's query:

                    Context:
                    {context}

                    User: {query}
                    AI:
                    """
            else:
                prompt = f"{prompt}\n\nUser query: {query}"
            print(prompt)
            return {**state, "prompted_query": prompt}
        elif url:
            web_content = scrape_wikipedia_page(url)
            chunks = split_document(web_content)
            # Take only first 4 documents
            docs = chunks[:4]
            if docs:
                # Use new modular vector store system
                vector_store = create_vector_store()
                ids = add_documents_to_store(vector_store, docs)
                retriever = create_retriever(vector_store, search_type="mmr", k=2)
                print(f"[Web Content] Using {len(docs)} chunks from scraped web content for context.")
                print(f"[Web Content] Added {len(ids)} documents to vector store")
                
                # Retrieve relevant docs using retriever
                relevant_docs = retriever.invoke(query)
                print(f"[Web Content] Retrieved {len(relevant_docs)} relevant chunks for context.")
                context = "\n".join([doc.page_content for doc in relevant_docs])
                prompt = f"""
                    You are an assistant with access to scraped web content. 
                    Use the following relevant context to answer the user's query:

                    Context:
                    {context}

                    User: {query}
                    AI:
                    """
            else:
                prompt = f"{prompt}\n\nUser query: {query}"
            print(prompt)
            return {**state, "prompted_query": prompt}

        return {**state, "prompted_query": f"{prompt}\n\nUser query: {query}"}

    def generate_response(state: AgentState) -> AgentState:
        llm = get_gemini()
        result = llm.invoke(state["prompted_query"])
        response_content = getattr(result, "content", str(result))
        return {**state, "response": response_content}

    print(f"Building graph with prompt type: {prompt_type}, URL: {url}, File Content: {bool(file_content)}")

    graph = StateGraph(AgentState)
    graph.add_node("Process Input", process_input)   # ✅ no url here
    graph.add_node("Generate Response", generate_response)

    graph.add_edge(START, "Process Input")
    graph.add_edge("Process Input", "Generate Response")
    graph.add_edge("Generate Response", END)

    return graph.compile()


# -------------------------
# Chat Agent
# -------------------------
def chat_agent(userQuery: str, prompt_type: str, url: str = None, file_content: str = None, thread_id: str = None) -> str:
    graph = build_graph(prompt_type, url, file_content)
    if thread_id is None:
        thread_id = "session-" + str(uuid.uuid4())

    state = graph.invoke({"query": userQuery})
    return state["response"]

