from pydantic import BaseModel, Field
from typing import List,Optional
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")


class EducationalExperience(BaseModel):
    """Educational Experience"""
    Institution: str = Field(..., description="The name of the institution")
    Degree: str = Field(..., description="The degree obtained")
    Field: str = Field(..., description="The field of study")
    Start_year: int 
    End_year: Optional[int]

class EducationalExperienceList(BaseModel):
    """List of Educational Experience"""
    edu_experiences: List[EducationalExperience]


tools=[EducationalExperienceList]

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    api_key=api_key
)

llm_with_tools = llm.bind_tools(tools)


def edu_exp(resume_text:str):
    print("Extracting educational experiences...")
    prompt = f"""Extract educational experiences from the resume text below according the tool:
Resume:
{resume_text}
"""
    response=llm_with_tools.invoke(prompt).tool_calls
    return response[0]['args']['edu_experiences']

