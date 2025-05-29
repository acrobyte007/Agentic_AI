from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from langchain_core.messages import AIMessage, HumanMessage
import uuid
from graph_c import graph
import json
import re

app = FastAPI()

# Pydantic models for request validation
class ResumeRequest(BaseModel):
    resume_text: str

class ResumeQuestionRequest(BaseModel):
    insights: str
    checkpoint_id: str

# Validate resume text format
def validate_resume_text(resume_text: str) -> bool:
    education_pattern = r"Education:\n\s*-\s*[^\n]+,\s*[^\n]+,\s*\d{4}-\d{4}"
    return bool(re.search(education_pattern, resume_text))

# Async generator for streaming summary and first question
async def stream_summary_and_question(resume_text: str, thread_id: str):
    # Log input for debugging
    print(f"Input resume_text:\n{resume_text}")
    
    # Validate resume text
    if not validate_resume_text(resume_text):
        yield f"data: Invalid resume text format. Expected education section like '- Degree, Institution, Start-End'\n\n"
        return
    
    initial_state = {
        "resume_text": [HumanMessage(content=resume_text)],
        "messages": [],
        "Work": [],
        "education": []
    }
    
    try:
        result = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})
        
        # Get summary from messages[-3] (after work, education, before insights)
        summary = result['messages'][-3].content if len(result['messages']) >= 3 else "Summary not found"
        yield f"data: {summary}\n\n"
        
        # Wait briefly to ensure connection stays open
        await asyncio.sleep(0.1)
        
        # Get the first question
        questions = result['messages'][-1].content
        try:
            if questions.startswith('[') and questions.endswith(']'):
                question_list = json.loads(questions)
                first_question = question_list[0] if question_list else "No questions generated"
            else:
                first_question = questions.split('\n')[0] if questions else "No questions generated"
        except json.JSONDecodeError:
            first_question = questions.split('\n')[0] if questions else "No questions generated"
        yield f"data: {first_question}\n\n"
        
        # Yield thread_id
        yield f"data: Thread ID: {thread_id}\n\n"
    
    except Exception as e:
        yield f"data: Error processing resume: {str(e)}\n\n"

@app.post("/analyze-resume")
async def analyze_resume(request: ResumeRequest):
    thread_id = str(uuid.uuid4())
    return StreamingResponse(
        stream_summary_and_question(request.resume_text, thread_id),
        media_type="text/event-stream",
        headers={"X-Thread-ID": thread_id}
    )

@app.post("/resume-question")
async def resume_question(request: ResumeQuestionRequest):
    try:
        checkpoint_id = request.checkpoint_id
        # Note: This endpoint doesn't use checkpointer in the provided graph
        # Simulate resuming at questions node
        resume_state = {
            "messages": [AIMessage(content=request.insights)],
            "Work": [],
            "education": [],
            "resume_text": []
        }
        
        result = await graph.ainvoke(
            resume_state,
            config={"configurable": {"thread_id": checkpoint_id}},
            from_node="questions"
        )
        
        questions = result['messages'][-1].content
        try:
            if questions.startswith('[') and questions.endswith(']'):
                question_list = json.loads(questions)
            else:
                question_list = questions.split('\n') if questions else []
        except json.JSONDecodeError:
            question_list = questions.split('\n') if questions else []
        
        return {"questions": question_list}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resuming workflow: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)