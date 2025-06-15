from uuid import uuid4
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
import json

from work_exp import work_experience
from educational_exp import edu_exp
from summary import summary_generator
from questions_generation import generate_questions

class State(TypedDict):
    messages: Annotated[list, add_messages]
    Work: Annotated[list, add_messages]
    education: Annotated[list, add_messages]
    resume_text: Annotated[list, add_messages]

def work_exp_generator(state: State):
    work_data = work_experience(state['resume_text'][-1].content)  # List[dict]
    return {"Work": [AIMessage(content=json.dumps(work_data))], "messages": [AIMessage(content=json.dumps(work_data))]}

def edu_exp_generator(state: State):
    education_data = edu_exp(state['resume_text'][-1].content)  # List[dict]
    return {"education": [AIMessage(content=json.dumps(education_data))], "messages": [AIMessage(content=json.dumps(education_data))]}

def makes_summary(state: State):
    work_content = json.loads(state['Work'][-1].content)  # List[dict]
    education_content = json.loads(state['education'][-1].content)  # List[dict]
    return {"messages": [AIMessage(content=summary_generator(json.dumps(work_content), json.dumps(education_content)))]}  # str

def questions_generator(state: State):
    questions = generate_questions(state['messages'][-1].content)  # List[str]
    return {"messages": [AIMessage(content="\n".join(questions or ["No questions generated"]))]}

workflow = StateGraph(State)
workflow.add_node("work_exp", work_exp_generator)
workflow.add_node("edu_exp", edu_exp_generator)
workflow.add_node("summary", makes_summary)
workflow.add_node("questions", questions_generator)
workflow.set_entry_point("work_exp")
workflow.add_edge("work_exp", "edu_exp")
workflow.add_edge("edu_exp", "summary")
workflow.add_edge("summary", "questions")
workflow.add_edge("questions", END)
graph = workflow.compile(checkpointer=InMemorySaver())

def analyze_resume(resume_text: str) -> dict:
    result = graph.invoke({
        "resume_text": [HumanMessage(content=resume_text)],
        "messages": [], "Work": [], "education": []
    }, {"configurable": {"thread_id": str(uuid4())}})
    return {
        "summary": result['messages'][-2].content,
        "work": json.loads(result['Work'][-1].content) if result['Work'] else [],
        "education": json.loads(result['education'][-1].content) if result['education'] else [],
        "questions": result['messages'][-1].content.split("\n")
    }

if __name__ == "__main__":
    sample_resume = """
    Work Experience:
    Software Engineer at TechCorp, 2020-2023: Developed Python-based web applications using Django.
    Data Scientist at DataInc, 2023-2025: Built machine learning models for predictive analytics.
    
    Education:
    B.S. in Computer Science, University of XYZ, 2016-2020.
    M.S. in Data Science, ABC University, 2021-2023.
    """
    result = analyze_resume(sample_resume)
    print("Summary:", result["summary"])
    print("Work:", result["work"])
    print("Education:", result["education"])
    print("Questions:", result["questions"])