from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from graph_n import analyze_resume, get_next_question, CHECKPOINTS, checkpointer, graph
import httpx
import asyncio
import threading
import logging
import traceback
from langchain_core.messages import AIMessage, HumanMessage
from uuid import uuid4
import re
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume Analysis API", description="API for analyzing resumes and paginating questions")

class ResumeRequest(BaseModel):
    resume_text: str

class ResumeQuestionResponse(BaseModel):
    question: Optional[str] = None
    message: Optional[str] = None

class ResumeQuestionRequest(BaseModel):
    checkpoint_id: str

class ResumeResumeRequest(BaseModel):
    checkpoint_id: str
    insights: Optional[str] = None

@app.post("/analyze-resume")
async def analyze_resume_endpoint(request: ResumeRequest):
    """
    Analyze a resume and stream the summary followed by the first question.
    """
    try:
        logger.info("Received /analyze-resume request")
        return await analyze_resume(request.resume_text)
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

@app.post("/resume-question", response_model=ResumeQuestionResponse)
async def resume_question_endpoint(request: ResumeQuestionRequest):
    """
    Get the next question for a given checkpoint ID.
    """
    try:
        logger.info(f"Received /resume-question request for checkpoint_id: {request.checkpoint_id}")
        result = get_next_question(request.checkpoint_id)
        if "error" in result:
            logger.error(f"Invalid checkpoint ID: {request.checkpoint_id}")
            raise HTTPException(status_code=404, detail=result["error"])
        return ResumeQuestionResponse(**result)
    except Exception as e:
        logger.error(f"Error fetching next question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching next question: {str(e)}")

@app.post("/resume-questions")
async def resume_questions_endpoint(request: ResumeResumeRequest):
    """
    Resume execution from the questions_generator node using a checkpoint ID.
    Optionally accepts insights to override the state.
    Returns the generated questions.
    """
    try:
        logger.info(f"Received /resume-questions request for checkpoint_id: {request.checkpoint_id}")
        
        if request.checkpoint_id not in CHECKPOINTS:
            logger.error(f"Invalid checkpoint ID: {request.checkpoint_id}")
            raise HTTPException(status_code=404, detail="Invalid checkpoint ID")

        checkpoint = checkpointer.get({"configurable": {"thread_id": request.checkpoint_id}})
        if not checkpoint:
            logger.error(f"No checkpoint found for ID: {request.checkpoint_id}")
            raise HTTPException(status_code=404, detail="No checkpoint found for ID")

        state = checkpoint["state"]
        
        if request.insights:
            logger.info("Using provided insights to override state")
            state["messages"] = [AIMessage(content=request.insights)]
        elif not state.get("messages"):
            logger.error("No insights available in state or request")
            raise HTTPException(status_code=400, detail="No insights available in state or request")

        config = {"configurable": {"thread_id": request.checkpoint_id}, "node": "questions"}
        result = await graph.ainvoke(state, config)

        questions_str = result["messages"][-1].content
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

        CHECKPOINTS[request.checkpoint_id]["questions"] = questions_list
        CHECKPOINTS[request.checkpoint_id]["current_question_index"] = 0

        return {"questions": questions_list}

    except Exception as e:
        logger.error(f"Error resuming questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resuming questions: {str(e)}")

async def test_workflow():
    """
    Test the workflow with a sample resume by running the stream_content generator.
    """
    sample_resume = """
    Software Engineer at TechCorp (2020-01 - Present): Developed web applications using Python.
    B.S. in Computer Science at State University (2014-2018)
    """
    logger.info("Testing analyze_resume with sample resume...")
    initial_state = {
        "resume_text": [HumanMessage(content=sample_resume)],
        "messages": [],
        "Work": [],
        "education": []
    }
    checkpoint_id = str(uuid4())
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
    
    async for chunk in stream_content():
        logger.info(f"Chunk: {chunk}")

async def test_resume_workflow():
    """
    Test resuming from the questions_generator node with a sample resume and checkpoint.
    """
    sample_resume = """
    Software Engineer at TechCorp (2020-01 - Present): Developed web applications using Python.
    Data Analyst at DataInc (2018-06 - 2019-12): Analyzed large datasets with SQL.
    B.S. in Computer Science at State University (2014-2018)
    """
    logger.info("Starting resume workflow test...")

    initial_state = {
        "resume_text": [HumanMessage(content=sample_resume)],
        "messages": [],
        "Work": [],
        "education": []
    }
    checkpoint_id = str(uuid4())
    config = {"configurable": {"thread_id": checkpoint_id}}
    
    result = await graph.ainvoke(initial_state, config)
    insights = result["messages"][-2].content
    logger.info(f"Generated insights: {insights}")

    logger.info(f"Resuming from questions node with checkpoint_id: {checkpoint_id}")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/resume-questions",
            json={"checkpoint_id": checkpoint_id, "insights": insights},
            timeout=30.0
        )
        logger.info(f"Resume response status: {response.status_code}")
        response_data = response.json()
        logger.info(f"Resumed questions: {response_data['questions']}")

def run_server():
    """
    Run the FastAPI server.
    """
    logger.info("Starting FastAPI server on http://localhost:8000")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    logger.info("Waiting for server to start (5 seconds)")
    asyncio.run(asyncio.sleep(5))
    
    logger.info("Running tests")
    asyncio.run(test_resume_workflow())