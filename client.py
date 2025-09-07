import streamlit as st
import requests  # Added for API calls
from langchain_community.document_loaders import WebBaseLoader  # Added for web loading

st.set_page_config(page_title="LLM Chat App", page_icon="🤖")

st.title("🤖 Simple LLM Chat App")

# ------------------------------
# Prompt Types
# ------------------------------
prompt_types = {
    "tot": "Tree of Thoughts",
    "cot": "Chain of Thoughts",
    "ReAct": "Reason + Act (alternate reasoning and actions)",
    "Reflexion": "Reflect on mistakes and improve iteratively",
    "Self-Consistency": "Generate 3 reasoning paths",
    "Debate": "Multiple agents argue",
    "Plan and Solve": "Create plan and execute",
    "PaR": "Program Aided Reasoning (write and execute code)"
}

with st.sidebar:
    selected_prompt = st.selectbox(
        "Select Prompt Type",
        list(prompt_types.keys()),
        format_func=lambda x: f"{x}: {prompt_types[x]}"
    )

    # ------------------------------
    # URL Input Section
    # ------------------------------
    url = st.text_input("Enter URL (optional)", placeholder="https://example.com")

    # ------------------------------
    # File Upload Section
    # ------------------------------
    uploaded_file = st.file_uploader("Upload a file (optional)", type=["txt", "md", "py", "json"])

# ------------------------------
# Chat State
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
    {"role": "assistant", "content": "Hello! 👋 How can I help you today?"}
    ]

# ------------------------------
# Display Chat History
# ------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ------------------------------
# User Input
# ------------------------------
if user_input := st.chat_input("Type your message..."):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Call the FastAPI server for LLM response
    try:
        data = {"userQuery": user_input, "prompt_type": selected_prompt}
        if url:
            # Load web content using WebBaseLoader
            loader = WebBaseLoader(web_paths=[url])
            docs = []
            for doc in loader.lazy_load():
                docs.append(doc)
            if docs:
                doc = docs[0]
                content = doc.page_content[:1000]
                data["content"] = content
            else:
                data["content"] = "Failed to load content"
        files = None
        if uploaded_file:
            files = {"file": uploaded_file}
        api_response = requests.post("http://localhost:8000/chat/", data=data, files=files)
        if api_response.status_code == 200:
            response = api_response.json()  # Assuming the response is a dict with the message
        else:
            # Handle non-200 responses, including 422 with details
            response = f"Error: {api_response.status_code} - {api_response.text}"
    except Exception as e:
        response = f"Error: {str(e)}"

    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
