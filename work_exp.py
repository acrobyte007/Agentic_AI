from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")


class WorkExperience(BaseModel):
    """Work Experience"""
    print("work experinece tool is called")
    company: str =Field(description="Company name")
    role:  str = Field(description="Role name")
    start_date: Optional[str] = Field(default=None, description="work starting date Format: YYYY-MM")
    end_date: Optional[str] = Field(default=None, description="Work ending date Format: YYYY-MM or 'Present'")
    description: str =Field(description="Description of the work experience")

class WorkExperienceList(BaseModel):
    """List of work experiences"""
    work_experiences: List[WorkExperience]


tools=[WorkExperienceList]

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    api_key=api_key
)

llm_with_tools = llm.bind_tools(tools)


def work_experience(resume_text:str):
    print("Inside Work Experience function ")
    prompt = f"""Extract work experiences from the resume text below according the tool:
Resume:
{resume_text}
"""
    response=llm_with_tools.invoke(prompt).tool_calls
    print("work experience response is gnerated")
    return response[0]['args']['work_experiences']


if __name__ =="__main__":
    resume_text = """John Doe
Work Experience:
- Software Engineer, TechCorp, 2020-2023: Developed web applications using Python and Django.
- Data Scientist, DataInc, 2023-2025: Built machine learning models for predictive analytics.
Education:
- B.S. Computer Science, University of Example, 2016-2020
"""
    print(work_experience(resume_text))