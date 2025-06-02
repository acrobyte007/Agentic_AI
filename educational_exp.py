from pydantic import BaseModel
from typing import List, Optional
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class EducationalExperience(BaseModel):
    """Educational Experience"""
    Institution: Optional[str] = None
    Degree: Optional[str] = None
    Field: Optional[str] = None
    Start_year: Optional[int] = None
    End_year: Optional[int] = None

class EducationalExperienceList(BaseModel):
    """List of Educational Experiences"""
    edu_experiences: List[EducationalExperience]

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
).with_structured_output(EducationalExperienceList)

def edu_exp(resume_text: str) -> List[dict]:
    """Extract educational experiences from resume text."""
    logger.info("Extracting educational experiences...")
    prompt = f"""Extract educational experiences from the resume text below:
Resume:
{resume_text}
"""
    try:
        result = groq_model.invoke(prompt)
        return [exp.model_dump(exclude_none=True) for exp in result.edu_experiences]
    except Exception as e:
        logger.error(f"Error extracting educational experiences: {str(e)}")
        return []

if __name__ == "__main__":
    resume_text = """John Doe
Work Experience:
- Software Engineer, TechCorp, 2020-2023: Developed web applications using Python and Django.
- Data Scientist, DataInc, 2023-2025: Built machine learning models for predictive analytics.
Education:
- B.S. Computer Science, University of Example, 2016-2020
"""
    result = edu_exp(resume_text)
    logger.info(f"Extracted educational experiences: {result}")
    print(result)