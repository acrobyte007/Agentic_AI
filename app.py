from fastapi import FastAPI
from pydantic import BaseModel
from graph_n import analyze_resume

app = FastAPI(title="Resume Analysis API", description="API for analyzing resumes and retrieving questions")

class ResumeRequest(BaseModel):
    resume_text: str

@app.post("/analyze-resume")
async def analyze_resume_endpoint(request: ResumeRequest):
    
    return analyze_resume(request.resume_text)

def run_server():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
