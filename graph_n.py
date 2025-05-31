from uuid import uuid4
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from fastapi.responses import StreamingResponse
import re
import json
import asyncio


from work_exp import work_experience
from educational_exp import edu_exp
from summary import summary_generator
from insight_extractor import extract_insights
from questions_generation import generate_questions


CHECKPOINTS = {}

class State(TypedDict):
    messages: Annotated[list, add_messages]
    Work: Annotated[list, add_messages]
    education: Annotated[list, add_messages]
    resume_text: Annotated[list, add_messages]

async def work_exp_generator(state: State):
    work_data = work_experience(state['resume_text'][-1].content)
    work_str = "\n".join(
        f"{job['role']} at {job['company']} ({job['start_date']} - {job['end_date']}): {job['description']}"
        for job in work_data
    )
    print(f"\n[work_exp_generator] Output:")
    print(f"Work: {work_str}")
    print(f"Messages: {work_str}")
    return {"Work": [AIMessage(content=work_str)], "messages": [AIMessage(content=work_str)]}

async def edu_exp_generator(state: State):
    resume_text = state['resume_text'][-1].content
    cleaned_text = "\n".join(line for line in resume_text.splitlines() if not line.strip().startswith('#'))
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text.strip())
    
    education_data = edu_exp(cleaned_text)
    print(f"[edu_exp_generator] Education data type: {type(education_data)}")
    print(f"[edu_exp_generator] Raw education data: {education_data}")
    
    if isinstance(education_data, list):
        education_str = "\n".join(
            f"{edu.get('Degree', 'Unknown Degree')} in {edu.get('Field', 'Unknown Field')} "
            f"at {edu.get('Institution', 'Unknown Institution')} "
            f"({edu.get('Start_year', 'Unknown')} - {edu.get('End_year', 'Unknown')})"
            for edu in education_data
            if edu.get('Degree') and edu.get('Institution') and edu.get('Field')
        ) or "No valid education entries found"
    else:
        education_str = str(education_data) if education_data else "No education data extracted"
    
    print(f"\n[edu_exp_generator] Output:")
    print(f"Education: {education_str}")
    print(f"Messages: {education_str}")
    return {"education": [AIMessage(content=education_str)], "messages": [AIMessage(content=education_str)]}

async def makes_summary(state: State):
    summary_chunks = []
    async for chunk in summary_generator(state['Work'][-1].content, state['education'][-1].content):
        summary_chunks.append(chunk)
    summary = "".join(summary_chunks)
    print(f"[makes_summary] Summary data type: {type(summary)}")
    print(f"\n[makes_summary] Output:")
    print(f"Messages: {summary}")
    return {"messages": [AIMessage(content=summary)]}

async def insight_extractor(state: State):
    insights = extract_insights(state['messages'][-1].content)
    print(f"\n[insight_extractor] Output:")
    print(f"Messages: {insights}")
    return {"messages": [AIMessage(content=insights)]}

async def questions_generator(state: State):
    questions = generate_questions(state['messages'][-1].content)
    if isinstance(questions, list):
        questions = "\n".join(questions)
    print(f"\n[questions_generator] Output:")
    print(f"Messages: {questions}")
    return {"messages": [AIMessage(content=questions)]}


workflow = StateGraph(State)
workflow.add_node("work_exp", work_exp_generator)
workflow.add_node("edu_exp", edu_exp_generator)
workflow.add_node("summary", makes_summary)
workflow.add_node("insights", insight_extractor)
workflow.add_node("questions", questions_generator)

workflow.set_entry_point("work_exp")
workflow.add_edge("work_exp", "edu_exp")
workflow.add_edge("edu_exp", "summary")
workflow.add_edge("summary", "insights")
workflow.add_edge("insights", "questions")
workflow.add_edge("questions", END)


checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

async def analyze_resume(resume_text: str) -> StreamingResponse:
    """
    Analyzes resume text, streams summary as chunks, and streams the first question.
    Returns a StreamingResponse for FastAPI.
    """
    initial_state = {
        "resume_text": [HumanMessage(content=resume_text)],
        "messages": [],
        "Work": [],
        "education": []
    }
    checkpoint_id = str(uuid4())
    print(checkpoint_id)
    config = {"configurable": {"thread_id": checkpoint_id}}

    async def stream_content():
        result = await graph.ainvoke(initial_state, config)

        
        summary = result['messages'][-3].content
        yield "Summary: "
        for i in range(0, len(summary), 50):  
            yield summary[i:i+50]
            await asyncio.sleep(0.1)

       
        questions_str = result['messages'][-1].content
        questions_list = []
        if questions_str:
            cleaned_questions = re.sub(
                r'^Here\s+are\s+the\s+tailored\s+interview\s+questions\s+based\s+on\s+the\s+resume\s+insights:\s*\n*\[\n',
                '',
                questions_str,
                flags=re.DOTALL | re.IGNORECASE
            ).rstrip(']\n')
            questions_list = [
                q.strip().strip('",') for q in cleaned_questions.split('\n')
                if q.strip() and q.strip('",') and not q.strip().startswith('[')
            ]

       
        if questions_list:
            yield f"\nFirst interview question: {questions_list[0]}"

        
        CHECKPOINTS[checkpoint_id] = {
            "summary": summary,
            "questions": questions_list,
            "current_question_index": 0
        }

    return StreamingResponse(stream_content(), media_type="text/plain")

def get_next_question(checkpoint_id: str) -> dict:
    """
    Returns the next question for a given checkpoint ID.
    """
    if checkpoint_id not in CHECKPOINTS:
        return {"error": "Invalid checkpoint ID"}

    state = CHECKPOINTS[checkpoint_id]
    current_index = state["current_question_index"]
    questions = state["questions"]

    next_index = current_index + 1
    CHECKPOINTS[checkpoint_id]["current_question_index"] = next_index

    if next_index >= len(questions):
        return {"message": "No more questions available"}

    return {
        "question": questions[next_index]
    }

if __name__ == "__main__":
    async def test_workflow():
        """
        Test the workflow with a sample resume by running the stream_content generator.
        """
        sample_resume = """
        Software Engineer at TechCorp (2020-01 - Present): Developed web applications using Python.
        B.S. in Computer Science at State University (2014-2018)
        """
        print("Testing analyze_resume with sample resume...")
        initial_state = {
            "resume_text": [HumanMessage(content=sample_resume)],
            "messages": [],
            "Work": [],
            "education": []
        }
        checkpoint_id = str(uuid4())
        config = {"configurable": {"thread_id": checkpoint_id}}
        
        # Run the workflow and stream content directly
        result = await graph.ainvoke(initial_state, config)
        async def stream_content():
            summary = result['messages'][-3].content
            yield "Summary: "
            for i in range(0, len(summary), 50):
                yield summary[i:i+50]
                await asyncio.sleep(0.1)
            questions_str = result['messages'][-1].content
            questions_list = []
            if questions_str:
                cleaned_questions = re.sub(
                    r'^Here\s+are\s+the\s+tailored\s+interview\s+questions\s+based\s+on\s+the\s+resume\s+insights:\s*\n*\[\n',
                    '',
                    questions_str,
                    flags=re.DOTALL | re.IGNORECASE
                ).rstrip(']\n')
                questions_list = [
                    q.strip().strip('",') for q in cleaned_questions.split('\n')
                    if q.strip() and q.strip('",') and not q.strip().startswith('[')
                ]
            if questions_list:
                yield f"\nFirst interview question: {questions_list[0]}"
            CHECKPOINTS[checkpoint_id] = {
                "summary": summary,
                "questions": questions_list,
                "current_question_index": 0
            }
        
        async for chunk in stream_content():
            print(f"Chunk: {chunk}")
        
    asyncio.run(test_workflow())