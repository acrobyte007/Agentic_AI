import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import List
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class InterviewQuestions(BaseModel):
    """Structured list of tailored interview questions"""
    questions: List[str]

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
).with_structured_output(InterviewQuestions)

def generate_questions(insights: str) -> List[str]:
    """Generate tailored interview questions based on resume insights."""
    logger.info("Generating interview questions...")
    if not insights.strip():
        logger.warning("Empty insights provided")
        return []
    
    prompt = f"""
    Based on the following resume insights, generate 10-12 tailored interview questions.
    Focus on technical skills, experiences, and soft skills highlighted in the insights.

    Resume Insights:
    {insights}
    """
    try:
        result = groq_model.invoke(prompt)
        return result.questions
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        return []

if __name__ == "__main__":
    sample_insights = """
    Proficient in Python and Django for web application development.
    Experienced in building machine learning models for predictive analytics.
    Strong teamwork skills demonstrated in collaborative projects.
    Effective problem-solving abilities in technical roles.
    """
    result = generate_questions(sample_insights)
    logger.info(f"Generated questions: {result}")
    print(result)