import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import List
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Insights(BaseModel):
    """Structured insights extracted from resume summary or data"""
    insights: List[str]

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set")

# Initialize Groq model with structured output
groq_model = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=api_key,
    max_retries=1
).with_structured_output(Insights)

def extract_insights(summary_or_data: str) -> List[str]:
    """Extract concise resume insights with technical topics and soft skills."""
    logger.info("Extracting insights...")
    prompt = f"""
    Extract concise resume insights from the following summary or structured resume data.
    Focus on technical expertise (e.g., programming languages, tools) and soft skills (e.g., teamwork, leadership).
    Return a list of insights as strings.

    Input:
    {summary_or_data}
    """
    try:
        result = groq_model.invoke(prompt)
        return result.insights
    except Exception as e:
        logger.error(f"Error extracting insights: {str(e)}")
        return []

if __name__ == "__main__":
    sample_input = """
    Summary: The candidate has a B.S. in Computer Science from University of Example (2016-2020) and experience including Software Engineer at TechCorp (2020-2023): Developed web applications using Python and Django. Data Scientist at DataInc (2023-2025): Built machine learning models for predictive analytics. Demonstrated strong teamwork and problem-solving skills.
    """
    result = extract_insights(sample_input)
    logger.info(f"Extracted insights: {result}")
    print(result)