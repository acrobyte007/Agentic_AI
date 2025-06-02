import os
import hashlib
import asyncio
from typing import AsyncGenerator
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Summary(BaseModel):
    """Structured summary of work experience and education"""
    summary: str

# Load environment variables
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY not set")

# Initialize Mistral model with structured output
llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    api_key=api_key
).with_structured_output(Summary)

SUMMARY_CACHE = {}

def _create_cache_key(work: str, education: str) -> str:
    """Create a unique cache key by hashing work and education inputs."""
    input_str = f"{work}||{education}"
    return hashlib.md5(input_str.encode('utf-8')).hexdigest()

async def summary_generator(work: str, education: str) -> AsyncGenerator[str, None]:
    """Generate a summary of work experience and education, with caching and streaming."""
    logger.info("Generating summary...")
    
    if not work or not education:
        yield "Error: Work or education input is empty."
        return

    cache_key = _create_cache_key(work, education)
    if cache_key in SUMMARY_CACHE:
        logger.info(f"Cache hit for key: {cache_key}")
        cached_summary = SUMMARY_CACHE[cache_key]
        yield cached_summary
        return

    prompt = f"""
    Generate a concise summary (100-150 words) of the following work experience and education.
    Focus on key technical skills, roles, and educational achievements.

    Work experience: {work}
    Education: {education}
    """
    try:
        result = await llm.ainvoke(prompt)
        summary = result.summary
        if len(SUMMARY_CACHE) < 1000:
            SUMMARY_CACHE[cache_key] = summary
            logger.info(f"Cached summary for key: {cache_key}")
        yield summary
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        yield f"Error: {str(e)}"

if __name__ == "__main__":
    async def test_summary_generator():
        """Test summary_generator with sample inputs, running twice to verify caching."""
        sample_work = """
        Software Engineer at TechCorp (2020-01 - Present): Developed web applications using Python and Django.
        Data Analyst at DataInc (2018-06 - 2019-12): Analyzed large datasets.
        """
        sample_education = """
        B.S. in Computer Science at State University (2014 - 2018)
        Coding Bootcamp at CodeAcademy (2017)
        """

        logger.info("=== First Run (Generate New Summary) ===")
        full_summary = []
        async for chunk in summary_generator(sample_work, sample_education):
            logger.info(f"Chunk: {chunk}")
            full_summary.append(chunk)
        logger.info(f"Full Summary: {''.join(full_summary)}")

        logger.info("=== Second Run (Use Cache) ===")
        full_summary = []
        async for chunk in summary_generator(sample_work, sample_education):
            logger.info(f"Chunk: {chunk}")
            full_summary.append(chunk)
        logger.info(f"Full Summary: {''.join(full_summary)}")

    asyncio.run(test_summary_generator())