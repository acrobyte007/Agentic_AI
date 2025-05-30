from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from graph_n import analyze_resume, get_next_question

app = FastAPI(title="Resume Analysis API", description="API for analyzing resumes and paginating questions")

# Request model for /analyze-resume
class ResumeRequest(BaseModel):
    resume_text: str

# Response model for /analyze-resume
class AnalyzeResumeResponse(BaseModel):
    checkpoint_id: str
    summary: str
    question: str

# Response model for /resume-question
class ResumeQuestionResponse(BaseModel):
    question: Optional[str] = None
    message: Optional[str] = None

# Request model for /resume-question
class ResumeQuestionRequest(BaseModel):
    checkpoint_id: str

@app.post("/analyze-resume", response_model=AnalyzeResumeResponse)
async def analyze_resume_endpoint(request: ResumeRequest):
    """
    Analyze a resume and return a summary with the first question.
    """
    try:
        result = analyze_resume(request.resume_text)
        return AnalyzeResumeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

@app.post("/resume-question", response_model=ResumeQuestionResponse)
async def resume_question_endpoint(request: ResumeQuestionRequest):
    """
    Get the next question for a given checkpoint ID.
    """
    result = get_next_question(request.checkpoint_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return ResumeQuestionResponse(**result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)