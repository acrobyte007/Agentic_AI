from pydantic import BaseModel
from typing import List, Optional
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import logging

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

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

groq_model = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=api_key,
    max_retries=3
).with_structured_output(EducationalExperienceList)

def edu_exp(resume_text: str) -> List[dict]:
    """Extract educational experiences from resume text."""
    logger.info("Extracting educational experiences...")
    prompt = f"""
Extract educational experiences from the following resume text and return them in a structured format with fields: Institution, Degree, Field, Start_year, End_year.

Resume:
{resume_text}

Return a list of educational experiences in the format:
{{
  "edu_experiences": [
    {{
      "Institution": str,
      "Degree": str,
      "Field": str,
      "Start_year": int,
      "End_year": int
    }}
  ]
}}
"""
    try:
        result = groq_model.invoke(prompt)
        return [exp.model_dump(exclude_none=True) for exp in result.edu_experiences]
    except Exception as e:
        logger.error(f"Error extracting educational experiences: {str(e)}")
        return []