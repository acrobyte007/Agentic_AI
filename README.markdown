# Resume Analysis API

## Overview

The **Acrobyte007** project is a FastAPI-based application designed to analyze resume text, generate a detailed summary, and produce tailored interview questions. It uses a LangGraph workflow to process resumes through nodes that extract work experience, education, generate summaries, derive insights, and create questions. The API provides two main endpoints:

- **POST `/analyze-resume`**: Processes a resume and streams the summary, first interview question, and a unique `checkpoint_id` as plain text, keeping the connection open.
- **POST `/resume-question`**: Retrieves the next question for a given `checkpoint_id`, enabling pagination through questions without reprocessing the resume.

State is managed in memory using a `CHECKPOINTS` dictionary, allowing efficient question pagination. The application leverages caching for summaries and integrates with external APIs (e.g., Mistral) for processing. Interactive API documentation is available via Swagger UI.

## Features

- **Resume Analysis**: Extracts work experience and education, generates a human-readable summary, and produces tailored interview questions.
- **Streaming Response**: Streams the summary and first question incrementally for `/analyze-resume`.
- **Question Pagination**: Serves questions one at a time using a `checkpoint_id`.
- **State Management**: Stores analysis results in memory for efficient retrieval.
- **Caching**: Caches summaries to reduce redundant API calls.
- **Error Handling**: Manages invalid inputs, missing `checkpoint_id`s, and processing errors.
- **Interactive Documentation**: Provides Swagger UI for testing endpoints.

## Project Structure

```
acrobyte007/
├── .gitignore                   # Git ignore file for excluding files like __pycache__
├── .env                         # Environment file for API keys (e.g., MISTRAL_API_KEY)
├── app.py                       # FastAPI application with endpoint definitions
├── graph_n.py                   # LangGraph workflow for resume processing
├── work_exp.py                  # Module for extracting work experience
├── educational_exp.py           # Module for extracting education details
├── summary.py                   # Module for generating and caching resume summaries
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
  - `langchain_mistralai`: For interacting with Mistral API
  - `python-dotenv`: For loading environment variables
  - `httpx`: For testing endpoints

- **API Key**: A valid `MISTRAL_API_KEY` stored in a `.env` file for Mistral API access.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/acrobyte007/Agentic_AI
   cd Agentic_AI
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```
   - On Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```

3. **Install Dependencies**:
   ```bash
   pip install fastapi uvicorn pydantic langchain langgraph langchain_mistralai python-dotenv httpx
   ```

4. **Set Up Environment**:
   - Create a `.env` file in the project root:
     ```bash
     echo "MISTRAL_API_KEY=api_key" > .env
     ```
   - Replace `api_key` with your Mistral API key.

5. **Verify Setup**:
   - Ensure all files (`app.py`, `graph_n.py`, `work_exp.py`, `educational_exp.py`, `summary.py`, `insight_extractor.py`, `questions_generation.py`, `.env`) are in `F:\Resume_Reviewer_Agent\Agentic_AI`.
   - Confirm Python 3.12 is active in the virtual environment:
     ```bash
     python --version
     ```

## Running the Application

1. **Start the FastAPI Server**:
   ```bash
   python app.py
   ```
   - This runs the API on `http://localhost:8000` and executes a test request to `/analyze-resume`.
   - Alternatively, run with Uvicorn directly:
     ```bash
     uvicorn app:app --host 0.0.0.0 --port 8000
     ```

2. **Access API Documentation**:
   - Open `http://localhost:8000/docs` in a browser to use Swagger UI.
   - Test endpoints via Swagger, curl, Postman, or a custom client.

## API Endpoints

### 1. POST `/analyze-resume`

**Description**: Analyzes resume text, streams a `checkpoint_id`, summary, and first interview question as plain text.

**Request Body**:
```json
{
  "resume_text": "string"
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/analyze-resume" -H "Content-Type: application/json" -d '{"resume_text": "Software Engineer at TechCorp (2020-01 - Present): Developed web applications using Python and Django.\nData Analyst at DataInc (2018-06 - 2019-12): Analyzed large datasets with SQL.\nB.S. in Computer Science at State University (2014-2018)"}'
```

**Response** (streamed text):
```
Checkpoint ID: 123e4567-e89b-12d3-a456-426614174000
Summary: **Summary of Work Experience and Education**

With a robust background in software engineering and data analysis, my professional journey has been marked by significant contributions to tech-driven organizations. My current role as a Software Engineer at TechCorp, which I have held since January 2020, has allowed me to develop and enhance web applications using Python and Django. This position has honed my skills in backend development, ensuring that the applications are not only functional but also scalable and efficient. My responsibilities include designing, coding, testing, and maintaining web applications, collaborating with cross-functional teams to define, design, and ship new features, and troubleshooting and debugging applications.

Prior to my current role, I served as a Data Analyst at DataInc from June 2018 to December 2019. In this capacity, I was responsible for analyzing large datasets using SQL. My work involved extracting meaningful insights from data, creating reports, and presenting findings to stakeholders. This experience sharpened my analytical skills and deepened my understanding of data-driven decision-making processes. I worked closely with various departments to identify trends, make data-driven recommendations, and improve business processes.

My educational foundation is anchored in a Bachelor of Science degree in Computer Science from State University, which I completed between 2014 and 2018. This comprehensive program equipped me with a strong theoretical and practical understanding of computer science principles. The curriculum covered a wide range of topics, including algorithms, data structures, software engineering, database management, and programming languages. This educational background has been instrumental in my ability to tackle complex technical challenges and develop innovative solutions.

Throughout my academic and professional career, I have consistently demonstrated a strong work ethic, attention to detail, and a commitment to continuous learning. My experience in both software engineering and data analysis has provided me with a unique perspective that allows me to approach problems from multiple angles. I am adept at working in team environments, as well as independently, and I thrive in fast-paced, dynamic settings.

In summary, my professional journey has been characterized by a blend of technical expertise and analytical prowess. My current role as a Software Engineer at TechCorp and my previous experience as a Data Analyst at DataInc, coupled with my educational background in Computer Science, have equipped me with a diverse skill set. This combination of experiences has prepared me to take on new challenges and contribute effectively to any organization.
First interview question: Can you walk me through a project where you optimized the scalability and efficiency of a backend application using Python and Django?
```

**Status Codes**:
- `200 OK`: Successful streaming response.
- `500 Internal Server Error`: Error processing the resume (e.g., invalid input or API failure).

### 2. POST `/resume-question`

**Description**: Retrieves the next interview question for a given `checkpoint_id`.

**Request Body**:
```json
{
  "checkpoint_id": "string"
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/resume-question" -H "Content-Type: application/json" -d '{"checkpoint_id": "123e4567-e89b-12d3-a456-426614174000"}'
```

**Response** (JSON):
```json
{
  "question": "How do you approach data analysis, and what tools do you use to extract insights and create reports?",
  "message": null
}
```

**Response (no more questions)**:
```json
{
  "question": null,
  "message": "No more questions available"
}
```

**Status Codes**:
- `200 OK`: Successful response with the next question or a message.
- `404 Not Found`: Invalid `checkpoint_id`.
- `500 Internal Server Error`: Unexpected server error.

## How It Works

1. **Resume Analysis** (`/analyze-resume`):
   - The resume text is processed via a LangGraph workflow in `graph_n.py` with nodes:
     - `work_exp_generator` (`work_exp.py`): Extracts work experience.
     - `edu_exp_generator` (`educational_exp.py`): Extracts education details.
     - `makes_summary` (`summary.py`): Generates a cached summary.
     - `insight_extractor` (`insight_extractor.py`): Derives insights from the summary.
     - `questions_generator` (`questions_generation.py`): Produces tailored interview questions.
   - A unique `checkpoint_id` is generated using `uuid4`.
   - The summary and first question are streamed, with the `checkpoint_id` included.
   - State (summary, questions, question index) is stored in `CHECKPOINTS`.

2. **Question Pagination** (`/resume-question`):
   - Retrieves the state from `CHECKPOINTS` using the `checkpoint_id`.
   - Increments the question index and returns the next question.
   - Returns “No more questions available” when exhausted.
   - Uses in-memory state, avoiding workflow re-execution.

3. **State Management**:
   - `CHECKPOINTS` in `graph_n.py` stores state in memory, keyed by `checkpoint_id`.
   - `InMemorySaver` from LangGraph ensures workflow consistency.

4. **Caching**:
   - `summary.py` caches summaries to reduce Mistral API calls, improving performance.

## Dependencies

- **Python Packages**:
  - `fastapi`
  - `uvicorn`
  - `pydantic`
  - `langchain`
  - `langgraph`
  - `langchain_mistralai`
  - `python-dotenv`
  - `httpx`

- **Included Modules**:
  - `work_exp.py`
  - `educational_exp.py`
  - `summary.py`
  - `insight_extractor.py`
  - `questions_generation.py`

## Error Handling

- **Invalid Resume Text**: Returns a 500 error if processing fails.
- **Invalid Checkpoint ID**: Returns a 404 error for `/resume-question`.
- **No Questions**: Returns “No more questions available” when questions are exhausted.
- **API Errors**: Logs Mistral API failures for debugging.

## Example Usage

1. **Analyze a Resume**:
   ```bash
   curl -X POST "http://localhost:8000/analyze-resume" -H "Content-Type: application/json" -d '{"resume_text": "Software Engineer at TechCorp (2020-01 - Present): Developed web applications using Python and Django.\nData Analyst at DataInc (2018-06 - 2019-12): Analyzed large datasets with SQL.\nB.S. in Computer Science at State University (2014-2018)"}'
   ```
   - Response (streamed):
     ```
     Checkpoint ID: 123e4567-e89b-12d3-a456-426614174000
     Summary: **Summary of Work Experience and Education**...
     First interview question: Can you walk me through a project where you optimized the scalability and efficiency of a backend application using Python and Django?
     ```

2. **Get Next Question**:
   ```bash
   curl -X POST "http://localhost:8000/resume-question" -H "Content-Type: application/json" -d '{"checkpoint_id": "123e4567-e89b-12d3-a456-426614174000"}'
   ```
   - Response:
     ```json
     {
       "question": "How do you approach data analysis, and what tools do you use to extract insights and create reports?",
       "message": null
     }
     ```
