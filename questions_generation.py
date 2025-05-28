import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    api_key=api_key
)

def generate_questions(insights: str) -> str:
    prompt = f"""Based on the following resume insights, generate a list of tailored interview questions:
Resume Insights:
{insights}

Return only the questions as a Python list of strings.
"""
    response = llm.invoke(prompt)
    return response.content  
