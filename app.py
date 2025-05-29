from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from langchain_core.messages import AIMessage, HumanMessage
import uuid
from graph_c import graph, checkpointer
import json

app = FastAPI()

# Pydantic models for request validation
class ResumeRequest(BaseModel):
    resume_text: str

class ResumeQuestionRequest(BaseModel):
    insights: str
    checkpoint_id: str

# Async generator for streaming summary and first question
async def stream_summary_and_question(resume_text: str, thread_id: str):
    initial_state = {
        "resume_text": [HumanMessage(content=resume_text)],
        "messages": [],
        "Work": [],
        "education": []
    }
    
    # Run the graph
    result = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})
    
    # Get education data as a fallback for summary (since makes_summary doesn't update messages)
    education = result['education'][-1].content if result['education'] else "Education data not found"
    yield f"data: {education}\n\n"
    
    # Wait briefly to ensure connection stays open
    await asyncio.sleep(0.1)
    
    # Get the first question
    questions = result['messages'][-1].content
    try:
        # Handle case where questions might be a JSON-like string
        if questions.startswith('[') and questions.endswith(']'):
            question_list = json.loads(questions)
            first_question = question_list[0] if question_list else "No questions generated"
        else:
            # Assume newline-separated questions
            first_question = questions.split('\n')[0] if questions else "No questions generated"
    except json.JSONDecodeError:
        # Fallback to splitting if JSON parsing fails
        first_question = questions.split('\n')[0] if questions else "No questions generated"
    yield f"data: {first_question}\n\n"
    
    # Yield thread_id for debugging
    yield f"data: Thread ID: {thread_id}\n\n"

@app.post("/analyze-resume")
async def analyze_resume(request: ResumeRequest):
    thread_id = str(uuid.uuid4())
    return StreamingResponse(
        stream_summary_and_question(request.resume_text, thread_id),
        media_type="text/event-stream",
        headers={"X-Thread-ID": thread_id}  # Return thread_id in header
    )

@app.post("/resume-question")
async def resume_question(request: ResumeQuestionRequest):
    try:
        # Validate checkpoint_id
        checkpoint_id = request.checkpoint_id
        # Load the saved state
        saved_state = checkpointer.get({"configurable": {"thread_id": checkpoint_id}})
        if not saved_state:
            raise HTTPException(status_code=404, detail="Invalid checkpoint ID")
        
        # Create a new state with provided insights
        resume_state = {
            "messages": [AIMessage(content=request.insights)],
            "Work": saved_state.get("Work", []),
            "education": saved_state.get("education", []),
            "resume_text": saved_state.get("resume_text", [])
        }
        
        # Resume from questions_generator node
        result = await graph.ainvoke(
            resume_state,
            config={"configurable": {"thread_id": checkpoint_id}},
            from_node="questions"  # Resume at questions node
        )
        
        # Parse questions
        questions = result['messages'][-1].content
        try:
            # Handle case where questions might be a JSON-like string
            if questions.startswith('[') and questions.endswith(']'):
                question_list = json.loads(questions)
            else:
                # Assume newline-separated questions
                question_list = questions.split('\n') if questions else []
        except json.JSONDecodeError:
            # Fallback to splitting
            question_list = questions.split('\n') if questions else []
        
        return {"questions": question_list}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resuming workflow: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)