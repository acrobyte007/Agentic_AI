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


def extract_insights(summary_or_data: str):
    print("Extracting insights...")
    prompt = f"""Extract concise resume insights from the following summary or structured resume data.
Return the insights in the following format with technical topics and soft skills:
{{
  "insights": [
    "example insight 1",
    "example insight 2"
  ]
}}

Input:
{summary_or_data}
"""
    response = groq_model.invoke(prompt)
   
    return response.content 
