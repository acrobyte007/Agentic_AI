import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

groq_model = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=api_key,
    max_retries=1
)

def generate_questions(insights: str) -> str:
    print("inside questions generation")
    prompt = f"""Based on the following resume insights, generate a list of tailored interview questions:
Resume Insights:
{insights}

Return only the questions as a Python list of strings.
"""
    response = groq_model.invoke(prompt)
    return response.content  
