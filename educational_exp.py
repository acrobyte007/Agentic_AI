from pydantic import BaseModel, Field
from typing import List,Optional
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


class EducationalExperience(BaseModel):
    """Educational Experience"""
    print("Educational Experience tool is called")
    Institution: Optional[str] = Field(..., description="The name of the institution")
    Degree: Optional[str] = Field(..., description="The degree obtained")
    Field: Optional[str] = Field(..., description="The field of study")
    Start_year: Optional[int] 
    End_year: Optional[int]

class EducationalExperienceList(BaseModel):
    """List of Educational Experience"""
    edu_experiences: List[EducationalExperience]


tools=[EducationalExperienceList]

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

groq_model = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=api_key,
    max_retries=1
)
llm_with_tools = groq_model.bind_tools(tools)

def edu_exp(resume_text:str):
    print("Extracting educational experiences...")
    prompt = f"""Extract educational experiences from the resume text below according the tool:
Resume:
{resume_text}
"""
    response=llm_with_tools.invoke(prompt).tool_calls
    return response[0]['args']['edu_experiences']

if __name__ =="__main__":
    resume_text = """John Doe
Work Experience:
- Software Engineer, TechCorp, 2020-2023: Developed web applications using Python and Django.
- Data Scientist, DataInc, 2023-2025: Built machine learning models for predictive analytics.
Education:
- B.S. Computer Science, University of Example, 2016-2020
"""
    print(edu_exp(resume_text))