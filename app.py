from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from graph_n import analyze_resume, CHECKPOINTS, checkpointer, graph
import threading
import logging
import re
from langchain_core.messages import AIMessage

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Thread lock for safe updates to CHECKPOINTS
checkpoints_lock = threading.Lock()

app = FastAPI(title="Resume Analysis API", description="API for analyzing resumes and retrieving questions")

class ResumeRequest(BaseModel):
    resume_text: str

class ResumeQuestionResponse(BaseModel):
    question: Optional[str] = None
    message: Optional[str] = None

class ResumeResumeRequest(BaseModel):
    checkpoint_id: str

@app.post("/analyze-resume")
async def analyze_resume_endpoint(request: ResumeRequest):
    """
    Analyze a resume and stream the summary, work, education, first question, and checkpoint ID.
    """
    try:
        logger.info("Received /analyze-resume request")
        return await analyze_resume(request.resume_text)
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

@app.post("/resume-questions", response_model=ResumeQuestionResponse)
async def resume_questions_endpoint(request: ResumeResumeRequest):
    """
    Retrieve the next question for the given checkpoint ID.
    Returns one question per request, incrementing the current question index.
    """
    try:
        logger.info(f"Received /resume-questions request for checkpoint_id: {request.checkpoint_id}")
        
        with checkpoints_lock:
            # Validate checkpoint ID
            if request.checkpoint_id not in CHECKPOINTS:
                logger.error(f"Invalid checkpoint ID: {request.checkpoint_id}")
                raise HTTPException(status_code=404, detail="Invalid checkpoint ID")

            # Retrieve checkpoint
            checkpoint = checkpointer.get({"configurable": {"thread_id": request.checkpoint_id}})
            if not checkpoint:
                logger.error(f"No checkpoint found for ID: {request.checkpoint_id}")
                raise HTTPException(status_code=404, detail="No checkpoint found for ID")

            # Log checkpoint for debugging
            logger.debug(f"Checkpoint content: {checkpoint}")

            # Access state (InMemorySaver stores state directly in checkpoint)
            state = checkpoint.get("values", checkpoint)
            if isinstance(state, dict) and "messages" not in state:
                logger.error(f"State missing 'messages' key: {state}")
                raise HTTPException(status_code=400, detail="Invalid state: No messages available")

            # Check if questions are already stored
            if "questions" not in CHECKPOINTS[request.checkpoint_id] or not CHECKPOINTS[request.checkpoint_id]["questions"]:
                # Generate questions if not already stored
                config = {"configurable": {"thread_id": request.checkpoint_id}}
                result = await graph.ainvoke(state, config, start_from_node="questions")

                questions_str = result["messages"][-1].content
                questions_list = []
                if questions_str:
                    # Clean and parse questions
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

                # Store questions and initialize index
                CHECKPOINTS[request.checkpoint_id]["questions"] = questions_list
                CHECKPOINTS[request.checkpoint_id]["current_question_index"] = 0

            # Get current question index and questions list
            current_index = CHECKPOINTS[request.checkpoint_id]["current_question_index"]
            questions_list = CHECKPOINTS[request.checkpoint_id]["questions"]

            # Handle no questions
            if not questions_list:
                logger.info(f"No questions available for checkpoint_id: {request.checkpoint_id}")
                return ResumeQuestionResponse(message="No questions available")

            # Handle end of questions
            if current_index >= len(questions_list):
                logger.info(f"No more questions available for checkpoint_id: {request.checkpoint_id}")
                return ResumeQuestionResponse(message="No more questions available")

            # Return next question and increment index
            next_question = questions_list[current_index]
            CHECKPOINTS[request.checkpoint_id]["current_question_index"] += 1

            logger.info(f"Returning question {current_index + 1}/{len(questions_list)}: {next_question}")
            return ResumeQuestionResponse(question=next_question)

    except Exception as e:
        logger.error(f"Error resuming questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resuming questions: {str(e)}")

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