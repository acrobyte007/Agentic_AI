from pydantic import BaseModel
from typing import List, Optional
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set")

class EducationalExperience(BaseModel):
    Institution: Optional[str]
    Degree: Optional[str]
    Field: Optional[str]
    Start_year: Optional[int]
    End_year: Optional[int]

class EducationalExperienceList(BaseModel):
    edu_experiences: List[EducationalExperience]

groq_model = ChatGroq(model_name="llama3-8b-8192", groq_api_key=api_key).with_structured_output(EducationalExperienceList)

def edu_exp(resume_text: str) -> List[dict]:
    """Extract educational experiences from resume text."""
    if not resume_text.strip():
        logger.warning("Empty resume text provided")
        return []
    
    prompt = f"""
Extract educational experiences from this resume, returning Institution, Degree, Field, Start_year, End_year for each:
{resume_text}
"""
    try:
        result = groq_model.invoke(prompt)
        return [exp.model_dump(exclude_none=True) for exp in result.edu_experiences]
    except Exception as e:
        logger.error(f"Error extracting educational experiences: {str(e)}")
        return []

if __name__ == "__main__":
    sample_resume = """
    Education:
    - University of XYZ, Bachelor of Science in Computer Science, 2018-2022
    - ABC Community College, Associate Degree in Mathematics, 2016-2018
    """
    experiences = edu_exp(sample_resume)
    print(experiences)