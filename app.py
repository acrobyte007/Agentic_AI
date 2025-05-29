from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from langchain_core.messages import AIMessage, HumanMessage
import uuid
from graph_c import graph  # assuming your graph is imported from graph_c.py
import json
import re

app = FastAPI()

# Request models
class ResumeRequest(BaseModel):
    resume_text: str

class ResumeQuestionRequest(BaseModel):
    insights: str
    checkpoint_id: str

# Validate the resume format
def validate_resume_text(resume_text: str) -> bool:
    education_pattern = r"Education:\n\s*-\s*[^\n]+,\s*[^\n]+,\s*\d{4}-\d{4}"
    return bool(re.search(education_pattern, resume_text))

# Streaming generator to send summary, first question, and follow-up questions
async def stream_summary_and_questions(resume_text: str, thread_id: str):
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
        # First graph invocation: full analysis
        result = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})

        # Get summary
        summary = result['messages'][-3].content if len(result['messages']) >= 3 else "Summary not found"
        yield f"data: Summary: {summary}\n\n"
        await asyncio.sleep(0.1)

        # Get first question
        questions_raw = result['messages'][-1].content
        try:
            if questions_raw.startswith('['):
                question_list = json.loads(questions_raw)
                first_question = question_list[0] if question_list else "No questions"
            else:
                first_question = questions_raw.split('\n')[0]
        except Exception:
            first_question = questions_raw.split('\n')[0] if questions_raw else "No questions"
        yield f"data: First Question: {first_question}\n\n"
        await asyncio.sleep(0.1)

        # Second graph invocation: resume from "questions"
        resume_state = {
            "messages": [AIMessage(content=summary)],
            "Work": [],
            "education": [],
            "resume_text": []
        }

        follow_up = await graph.ainvoke(
            resume_state,
            config={"configurable": {"thread_id": thread_id}},
            from_node="questions"
        )

        follow_up_questions = follow_up['messages'][-1].content
        try:
            if follow_up_questions.startswith('['):
                questions = json.loads(follow_up_questions)
            else:
                questions = follow_up_questions.split('\n')
        except Exception:
            questions = follow_up_questions.split('\n')

        for q in questions:
            yield f"data: Follow-up Question: {q}\n\n"
        yield f"data: Thread ID: {thread_id}\n\n"

    except Exception as e:
        yield f"data: Error: {str(e)}\n\n"

# Analyze and stream
@app.post("/analyze-resume")
async def analyze_resume(request: ResumeRequest):
    thread_id = str(uuid.uuid4())
    return StreamingResponse(
        stream_summary_and_questions(request.resume_text, thread_id),
        media_type="text/event-stream",
        headers={"X-Thread-ID": thread_id}
    )

# Resume from questions separately if needed
@app.post("/resume-question")
async def resume_question(request: ResumeQuestionRequest):
    try:
        resume_state = {
            "messages": [AIMessage(content=request.insights)],
            "Work": [],
            "education": [],
            "resume_text": []
        }

        result = await graph.ainvoke(
            resume_state,
            config={"configurable": {"thread_id": request.checkpoint_id}},
            from_node="questions"
        )

        questions_raw = result['messages'][-1].content
        try:
            if questions_raw.startswith('['):
                questions = json.loads(questions_raw)
            else:
                questions = questions_raw.split('\n')
        except Exception:
            questions = questions_raw.split('\n')

        return {"questions": questions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
