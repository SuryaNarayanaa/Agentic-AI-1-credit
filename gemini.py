# pip install langchain-google-genai python-dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Set up Google Gemini client
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

def get_prompt(bot_name, prompt, prog, student):

    # 1. Create a Multi-Message Prompt
    chat_template = ChatPromptTemplate.from_messages([
        ("system", f"You are {bot_name}, a helpful and knowledgeable AI assistant specializing in {prog} programming, and you are tasked to teach for {student} students."),
        ("human", "{user_input}"),
    ])
    
    # 2. Format the prompt messages defined in the template
    # This returns a list of chat messages (AIMessage, SystemMessage, HumanMessage) ready for the model.
    chat_prompt = chat_template.format_messages(bot_name = bot_name, user_input = prompt)
    print(chat_prompt)

    return prompt

def get_gemini():
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
        # other params...
    )
    return llm

if __name__ == "__main__":
    bot_name = "lysa"
    prog =  "C#"
    prompt = "how OSU was made in C#"
    llm = get_gemini()
    student = "advanced"
    prompt = get_prompt(bot_name, prompt, prog, student)
    response = llm.invoke(prompt)
    print(f"Gemini Response:{response.content}\n")
    print(f"Token usage : {response.usage_metadata}\n")
    print(f"Full Response : {response}\n")