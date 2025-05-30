# Acrobyte007 - Resume Analysis API

## Overview

The **Acrobyte007** project is a FastAPI-based application designed to analyze resume text and generate a summary along with tailored interview questions. It leverages a LangGraph workflow to process resumes through a series of nodes that extract work experience, education, insights, and questions. The API provides two main endpoints:

- **`/analyze-resume`**: Processes a resume, generates a summary and a list of interview questions, and returns the first question with a unique checkpoint ID.
- **`/resume-question`**: Retrieves the next question for a given checkpoint ID, enabling pagination through the questions without reprocessing the resume.

State is managed in memory using a `CHECKPOINTS` dictionary, allowing efficient question pagination across multiple API calls. The application is designed for scalability and ease of use, with interactive API documentation via Swagger UI.

## Features

- **Resume Analysis**: Extracts work experience and education, generates a human-readable summary, and produces tailored interview questions.
- **Question Pagination**: Serves questions one at a time using a checkpoint ID, avoiding redundant processing.
- **State Management**: Stores analysis results in memory for efficient retrieval.
- **Error Handling**: Handles invalid inputs, missing checkpoint IDs, and processing errors.
- **Interactive Documentation**: Provides Swagger UI for testing endpoints.

## Project Structure

```
acrobyte007/
├── .gitignore                   # Git ignore file for excluding files like __pycache__
├── __pycache__/                 # Python cache directory (excluded from version control)
├── app.py                       # FastAPI application with endpoint definitions
├── graph_n.py                   # Core LangGraph workflow for resume processing
├── work_exp.py                  # Module for extracting work experience
├── educational_exp.py           # Module for extracting education details
├── summary.py                   # Module for generating resume summaries
├── insight_extractor.py         # Module for extracting insights from summaries
├── questions_generation.py      # Module for generating interview questions
└── README.md                    # Project documentation
```

## Prerequisites

- **Python**: Version 3.12 
- **Dependencies**:
  - `fastapi`: For building the API
  - `uvicorn`: For running the FastAPI server
  - `pydantic`: For request/response validation
  - `langchain`: For handling messages in the LangGraph workflow
  - `langgraph`: For defining the processing workflow

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/acrobyte007/Agentic_AI
   cd Agentic_AI
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

3. **Install Dependencies**:
   ```bash
   pip install fastapi uvicorn pydantic langchain langgraph
   ```

   Note: The modules `work_exp.py`, `educational_exp.py`, `summary.py`, `insight_extractor.py`, and `questions_generation.py` are included in the repository and do not require separate installation. Ensure they are present in the project directory.

4. **Verify Setup**:
   Confirm that all files (`app.py`, `graph_n.py`, `work_exp.py`, `educational_exp.py`, `summary.py`, `insight_extractor.py`, `questions_generation.py`) are in the project directory.

## Running the Application

1. **Start the FastAPI Server**:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

   This runs the API on `http://localhost:8000`.

2. **Access API Documentation**:
   - Open `http://localhost:8000/docs` in a browser to access the Swagger UI.
   - Use the UI to test endpoints or make requests via `curl`, Postman, or a custom client.

## API Endpoints

### 1. POST `/analyze-resume`

**Description**: Analyzes a resume text, generates a summary and a list of interview questions, and returns the first question with a checkpoint ID for pagination.

**Request Body**:
```json
{
    "resume_text": "string"
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/analyze-resume" -H "Content-Type: application/json" -d '{
    "resume_text": "John Doe\nWork Experience:\n- AI Developer, TechCorp, 2020-2023: Developed AI blog writer backend using LangChain and FastAPI.\nEducation:\n- B.S. Mechanical Engineering, University of Example, 2016-2020\n- M.Tech Artificial Intelligence, Tech Institute, 2023-Present"
}'
```

**Response**:
```json
{
    "checkpoint_id": "fb882939-f42e-486d-ba2c-42ccbee567a7",
    "summary": "Candidate has a strong educational background in Mechanical Engineering and is currently advancing in Artificial Intelligence (CSE). Proficient in AI development with experience in creating AI blog writer backend systems. Skilled in using advanced AI frameworks and tools such as LangChain, LangGraph, FastAPI, and React agents. Demonstrates expertise in integrating AI technologies to develop innovative solutions.",
    "question": "Can you walk me through your experience with integrating advanced AI frameworks and tools? How have you applied them in your previous roles?"
}
```

**Status Codes**:
- `200 OK`: Successful response with summary and first question.
- `500 Internal Server Error`: Error processing the resume.

### 2. POST `/resume-question`

**Description**: Retrieves the next interview question for a given checkpoint ID.

**Request Body**:
```json
{
    "checkpoint_id": "string"
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/resume-question" -H "Content-Type: application/json" -d '{
    "checkpoint_id": "fb882939-f42e-486d-ba2c-42ccbee567a7"
}'
```

**Response**:
```json
{
    "question": "How do you stay up-to-date with the latest advancements in AI and machine learning? Can you give me an example of a new concept you've learned recently?",
    "message": null
}
```

**Response (when no more questions)**:
```json
{
    "question": null,
    "message": "No more questions available"
}
```

**Status Codes**:
- `200 OK`: Successful response with the next question or a message indicating no more questions.
- `404 Not Found`: Invalid checkpoint ID.

## How It Works

1. **Resume Analysis** (`/analyze-resume`):
   - The resume text is processed through a LangGraph workflow defined in `graph_n.py` with the following nodes:
     - `work_exp_generator` (from `work_exp.py`): Extracts work experience.
     - `edu_exp_generator` (from `educational_exp.py`): Extracts education details.
     - `makes_summary` (from `summary.py`): Generates a resume summary.
     - `insight_extractor` (from `insight_extractor.py`): Extracts insights from the summary.
     - `questions_generator` (from `questions_generation.py`): Generates tailored interview questions.
   - The workflow produces a human-readable summary and a list of questions.
   - A unique checkpoint ID is generated using `uuid4`.
   - The summary, questions, and current question index (starting at 0) are stored in the `CHECKPOINTS` dictionary.
   - The response includes the checkpoint ID, summary, and the first question.

2. **Question Pagination** (`/resume-question`):
   - Retrieves the stored state using the checkpoint ID.
   - Increments the current question index and returns the next question.
   - If no more questions are available, returns a message indicating so.
   - Does not re-run the LangGraph workflow, ensuring efficient pagination.

3. **State Management**:
   - The `CHECKPOINTS` dictionary in `graph_n.py` stores state in memory, keyed by checkpoint ID.
   - Each entry contains the summary, list of questions, and current question index.
   - The `InMemorySaver` from LangGraph persists the workflow state for consistency.

## Dependencies

The application relies on the following Python packages:

- `fastapi`: For building the API.
- `uvicorn`: For running the FastAPI server.
- `pydantic`: For request/response model validation.
- `langchain`: For handling messages in the workflow.
- `langgraph`: For defining the processing workflow.

The following modules are included in the repository and do not require separate installation:
- `work_exp.py`
- `educational_exp.py`
- `summary.py`
- `insight_extractor.py`
- `questions_generation.py`

## Error Handling

- **Invalid Resume Text**: Returns a 500 error with a message if the resume cannot be processed.
- **Invalid Checkpoint ID**: Returns a 404 error for `/resume-question` if the checkpoint ID is not found.
- **Empty Questions**: Returns "No questions generated" if no questions are produced by the workflow.
- **Malformed Questions**: The `questions_generator` output is parsed to remove the preamble and extract individual questions, handling variations in format.

## Production Considerations

- **Persistent Storage**: Replace the in-memory `CHECKPOINTS` dictionary with a database (e.g., Redis, MongoDB) for scalability and persistence.
- **Authentication**: Add OAuth2 or JWT authentication to secure endpoints and associate checkpoints with users.
- **Rate Limiting**: Implement rate limiting to prevent abuse.
- **Logging**: Add logging for monitoring and debugging.
- **Deployment**: Use a production-grade server like Gunicorn with Uvicorn workers for deployment.
- **Error Handling**: Enhance error handling for malformed resume text or unexpected workflow outputs.

## Example Usage

1. **Analyze a Resume**:
   ```bash
   curl -X POST "http://localhost:8000/analyze-resume" -H "Content-Type: application/json" -d '{
       "resume_text": "John Doe\nWork Experience:\n- AI Developer, TechCorp, 2020-2023: Developed AI blog writer backend using LangChain and FastAPI.\nEducation:\n- B.S. Mechanical Engineering, University of Example, 2016-2020\n- M.Tech Artificial Intelligence, Tech Institute, 2023-Present"
   }'
   ```

   **Response**:
   ```json
   {
       "checkpoint_id": "fb882939-f42e-486d-ba2c-42ccbee567a7",
       "summary": "Candidate has a strong educational background in Mechanical Engineering and is currently advancing in Artificial Intelligence (CSE). Proficient in AI development with experience in creating AI blog writer backend systems. Skilled in using advanced AI frameworks and tools such as LangChain, LangGraph, FastAPI, and React agents. Demonstrates expertise in integrating AI technologies to develop innovative solutions.",
       "question": "Can you walk me through your experience with integrating advanced AI frameworks and tools? How have you applied them in your previous roles?"
   }
   ```

2. **Get Next Question**:
   ```bash
   curl -X POST "http://localhost:8000/resume-question" -H "Content-Type: application/json" -d '{
       "checkpoint_id": "fb882939-f42e-486d-ba2c-42ccbee567a7"
   }'
   ```

   **Response**:
   ```json
   {
       "question": "How do you stay up-to-date with the latest advancements in AI and machine learning? Can you give me an example of a new concept you've learned recently?",
       "message": null
   }
   ```

