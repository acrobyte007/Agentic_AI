import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

# Load API key
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# Initialize Mistral LLM
llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    api_key=api_key
)

# Insight extractor function
def extract_insights(summary_or_data: str):
    prompt = f"""Extract concise resume insights from the following summary or structured resume data.
Return the insights in the following format:
{{
  "insights": [
    "example insight 1",
    "example insight 2"
  ]
}}

Input:
{summary_or_data}
"""
    response = llm.invoke(prompt)
    return response.content
