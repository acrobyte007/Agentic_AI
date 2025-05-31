from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from graph_n import analyze_resume, get_next_question
import httpx
import asyncio
import threading
import logging
import traceback


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

if __name__ == "__main__":
    import uvicorn
    
    async def test_endpoint():
        """
        Test the /analyze-resume endpoint with a sample resume.
        """
        sample_resume = """
        Software Engineer at TechCorp (2020-01 - Present): Developed web applications using Python and Django.
        Data Analyst at DataInc (2018-06 - 2019-12): Analyzed large datasets with SQL.
        B.S. in Computer Science at State University (2014-2018)
        """
        logger.info("Starting /analyze-resume endpoint test")
        try:
            async with httpx.AsyncClient() as client:
                logger.info("Sending POST request to /analyze-resume")
                response = await client.post(
                    "http://localhost:8000/analyze-resume",
                    json={"resume_text": sample_resume},
                    timeout=30.0
                )
                logger.info(f"Received response with status: {response.status_code}")
                async for chunk in response.aiter_text():
                    print(f"Received chunk: {chunk}")
                    logger.info(f"Streamed chunk: {chunk}")
        except Exception as e:
            logger.error("Test failed")
            print(f"Test failed: {str(e)}")
            traceback.print_exc()

   
    def run_server():
        logger.info("Starting FastAPI server on http://localhost:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    
    logger.info("Waiting for server to start (5 seconds)")
    asyncio.run(asyncio.sleep(5))
    
    logger.info("Running endpoint test")
    asyncio.run(test_endpoint())