import os
import logging
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Summary(BaseModel):
    """Structured summary of work experience and education"""
    summary: str

# Load environment variables
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY not set")

# Initialize Mistral model with structured output
llm = ChatMistralAI(model="mistral-large-latest", temperature=0, api_key=api_key).with_structured_output(Summary)

def summary_generator(work: str, education: str) -> str:
    """Generate a summary of work experience and education."""
    if not (work.strip() or education.strip()):
        logger.warning("Empty work or education provided")
        return "No input provided."
    
    prompt = f"""
    Generate a concise summary (250 words) of the following work experience and education.
    Focus on key technical skills, roles, and educational achievements.

    Work experience: {work}
    Education: {education}
    """
    try:
        result = llm.invoke(prompt)
        return result.summary
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    sample_work = """
    Software Engineer at TechCorp, 2020-2023, developed Python-based web applications using Django.
    Data Scientist at DataInc, 2023-2025, built machine learning models for predictive analytics.
    """
    sample_education = """
    B.S. in Computer Science, University of XYZ, 2016-2020.
    M.S. in Data Science, ABC University, 2021-2023.
    """
    summary = summary_generator(sample_work, sample_education)
    print("Summary:", summary)