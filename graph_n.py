from uuid import uuid4
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from fastapi.responses import StreamingResponse
import asyncio
import logging
import re

# Imports for external functions
from work_exp import work_experience
from educational_exp import edu_exp
from summary import summary_generator
from insight_extractor import extract_insights
from questions_generation import generate_questions

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Checkpoint storage
CHECKPOINTS = {}

class State(TypedDict):
    messages: Annotated[list, add_messages]
    Work: Annotated[list, add_messages]
    education: Annotated[list, add_messages]
    resume_text: Annotated[list, add_messages]

async def work_exp_generator(state: State):
    work_data = work_experience(state['resume_text'][-1].content)
    work_str = "\n".join(
        f"{job['role']} at {job.get('company', 'Unknown')} ({job.get('start_date', 'Unknown')} - {job.get('end_date', 'Unknown')}): {job.get('description', '')}"
        for job in work_data
    ) or "No work experience extracted"
    logger.info(f"[work_exp_generator] Output: {work_str}")
    return {"Work": [AIMessage(content=work_str)], "messages": [AIMessage(content=work_str)]}

async def edu_exp_generator(state: State):
    resume_text = state['resume_text'][-1].content
    cleaned_text = "\n".join(line for line in resume_text.splitlines() if not line.strip().startswith('#'))
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text.strip())
    
    education_data = edu_exp(cleaned_text)
    logger.info(f"[edu_exp_generator] Education data type: {type(education_data)}")
    logger.info(f"[edu_exp_generator] Raw education data: {education_data}")
    
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
    
    logger.info(f"[edu_exp_generator] Output: {education_str}")
    return {"education": [AIMessage(content=education_str)], "messages": [AIMessage(content=education_str)]}

async def makes_summary(state: State):
    summary_chunks = []
    async for chunk in summary_generator(state['Work'][-1].content, state['education'][-1].content):
        summary_chunks.append(chunk)
    summary = "".join(summary_chunks)
    logger.info(f"[makes_summary] Summary data type: {type(summary)}")
    logger.info(f"[makes_summary] Output: {summary}")
    return {"messages": [AIMessage(content=summary)]}

async def insight_extractor(state: State):
    insights = extract_insights(state['messages'][-1].content)
    insights_str = "\n".join(insights) if insights else "No insights extracted"
    logger.info(f"[insight_extractor] Output: {insights_str}")
    return {"messages": [AIMessage(content=insights_str)]}

async def questions_generator(state: State):
    questions = generate_questions(state['messages'][-1].content)
    questions_str = "\n".join(questions) if questions else "No questions generated"
    logger.info(f"[questions_generator] Output: {questions_str}")
    return {"messages": [AIMessage(content=questions_str)]}

# Workflow setup
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
    initial_state = {
        "resume_text": [HumanMessage(content=resume_text)],
        "messages": [],
        "Work": [],
        "education": []
    }
    checkpoint_id = str(uuid4())
    logger.info(f"Generated checkpoint_id: {checkpoint_id}")
    config = {"configurable": {"thread_id": checkpoint_id}}

    async def stream_content():
        result = await graph.ainvoke(initial_state, config)
        summary = result['messages'][-3].content  # Summary from makes_summary
        yield "Summary: "
        for i in range(0, len(summary), 50):
            yield summary[i:i+50]
            await asyncio.sleep(0.1)
        questions = result['messages'][-1].content.split("\n") if result['messages'][-1].content else []
        if questions and questions[0]:
            yield f"\nFirst interview question: {questions[0]}"
        CHECKPOINTS[checkpoint_id] = {
            "summary": summary,
            "questions": questions,
            "current_question_index": 0
        }
    
    headers = {"check_point_id": checkpoint_id}
    return StreamingResponse(stream_content(), media_type="text/plain", headers=headers)

def get_next_question(checkpoint_id: str) -> dict:
    if checkpoint_id not in CHECKPOINTS:
        return {"error": "Invalid checkpoint ID"}
    state = CHECKPOINTS[checkpoint_id]
    current_index = state["current_question_index"]
    questions = state["questions"]
    next_index = current_index + 1
    CHECKPOINTS[checkpoint_id]["current_question_index"] = next_index
    if next_index >= len(questions):
        return {"message": "No more questions available"}
    return {"question": questions[next_index]}

if __name__ == "__main__":
    async def test_workflow():
        sample_resume = """
        Software Engineer at TechCorp (2020-01 - Present): Developed web applications using Python.
        B.S. in Computer Science at State University (2014-2018)
        """
        logger.info("Testing analyze_resume with sample resume...")
        response = await analyze_resume(sample_resume)
        async for chunk in response.body_iterator:
            logger.info(f"Chunk: {chunk}")
    
    asyncio.run(test_workflow())