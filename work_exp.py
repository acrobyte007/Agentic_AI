from pydantic import BaseModel
from typing import Optional, List
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class WorkExperience(BaseModel):
    """Work Experience"""
    company: Optional[str] = None
    role: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None

class WorkExperienceList(BaseModel):
    """List of Work Experiences"""
    work_experiences: List[WorkExperience]

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    api_key=api_key
).with_structured_output(WorkExperienceList)

def work_experience(resume_text: str) -> List[dict]:
    """Extract work experiences from resume text."""
    prompt = f"""Extract work experiences from the resume text below. Include company, role, start date (YYYY-MM), end date (YYYY-MM or 'Present'), and description for each experience.

Resume:
{resume_text}
"""
    result = llm.invoke(prompt)
    return [exp.model_dump(exclude_none=True) for exp in result.work_experiences]

if __name__ == "__main__":
    resume_text = """John Doe
Work Experience:
- Software Engineer, TechCorp, 2020-2023: Developed web applications using Python and Django.
- Data Scientist, DataInc, 2023-2025: Built machine learning models for predictive analytics.
Education:
- B.S. Computer Science, University of Example, 2016-2020
"""
    result = work_experience(resume_text)
    print(result)