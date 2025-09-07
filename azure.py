import os
from openai import AzureOpenAI

import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Set up Azure OpenAI client
subscription_key = os.getenv("AZURE_OPENAI_KEY")    
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

def get_azure_response(prompt):

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": f"{prompt}",
            }
        ],
        max_completion_tokens=20,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=deployment
    )
    return response

if __name__ == "__main__":
    prompt = "What is agentic AI"
    response =get_azure_response(prompt)

    print("Full Response:", response)

    # Print the content (assistant reply)
    print("Content:", response.choices[0].message.content)

    # Print token usage
    print("Prompt tokens:", response.usage.prompt_tokens)
    print("Completion tokens:", response.usage.completion_tokens)
    print("Total tokens:", response.usage.total_tokens)