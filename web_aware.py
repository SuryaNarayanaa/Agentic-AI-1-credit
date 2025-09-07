import os
import uuid
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
from bs4 import BeautifulSoup


def scrape_wikipedia_page(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = soup.find('h1', id='firstHeading').text if soup.find('h1', id='firstHeading') else 'No title found'
        
        # Extract main content (simplified)
        content = soup.find('div', id='mw-content-text')
        paragraphs = content.find_all('p') if content else []
        text = '\n'.join([p.get_text(strip=True) for p in paragraphs])  # Clean text

        print(f"Scraped {len(paragraphs)} paragraphs from the page.")
        
        return f"Title: {title}\n\nContent:\n{text}"
    else:
        return f"Failed to retrieve the page. Status code: {response.status_code}"

# # Load environment variables
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# url = "https://en.wikipedia.org/wiki/Brutalist_architecture"
# scraped = scrape_wikipedia_page(url)
# # print(scraped)
# # -------------------------
# # Gemini LLM Setup
# # -------------------------
# def get_gemini():
#     if not GOOGLE_API_KEY:
#         raise ValueError("Missing GEMINI_API_KEY in environment variables.")
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash-lite",
#         temperature=0.3,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#         top_k=None,
#         top_p=None,
#         safety_settings=None,
#         api_key=GOOGLE_API_KEY,
#     )
#     return llm
# llm = get_gemini()
# # Simple chat loop
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ['exit', 'quit']:
#         break
#     prompt = f"""
# You are an assistant with access to some scraped information. 
# Always prioritize answering based on the content provided below. 
# If the scraped text doesn’t contain enough information, you may provide helpful background knowledge, 
# but clearly indicate when you are going beyond the scraped content.

# Scraped Data:
# \"\"\"{scraped}\"\"\"

# User: {user_input}
# AI:
# """


#     response = llm.invoke(prompt)
#     print(f"AI: {response.content}")