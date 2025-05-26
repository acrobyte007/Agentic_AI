from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")


class WorkExperience(BaseModel):
    """Work Experience"""
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


def work_exp(resume_text:str):
    prompt = f"""Extract work experiences from the resume text below according the tool:
Resume:
{resume_text}
"""
    response=llm_with_tools.invoke(prompt).tool_calls
    return response[0]['args']['work_experiences']

