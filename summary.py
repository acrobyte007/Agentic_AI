import os
import hashlib
import asyncio
from typing import AsyncGenerator
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")


llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    api_key=api_key
)


SUMMARY_CACHE = {}

def _create_cache_key(work: str, education: str) -> str:
    """
    Create a unique cache key by hashing the work and education inputs.
    """
    input_str = f"{work}||{education}"
    return hashlib.md5(input_str.encode('utf-8')).hexdigest()

async def summary_generator(work: str, education: str) -> AsyncGenerator[str, None]:
    """
    Generate a summary of work experience and education, with caching and streaming.
    Yields chunks of the summary for streaming to the client.
    
    Args:
        work (str): Work experience data as a string.
        education (str): Education data as a string.
    
    Yields:
        str: Chunks of the generated summary.
    """
    print("Inside summary generator")

    
    if not work or not education:
        yield "Error: Work or education input is empty."
        return

    
    cache_key = _create_cache_key(work, education)

  
    if cache_key in SUMMARY_CACHE:
        print(f"Cache hit for key: {cache_key}")
        cached_summary = SUMMARY_CACHE[cache_key]
       
        for i in range(0, len(cached_summary), 50):
            yield cached_summary[i:i+50]
        return

    # Generate new summary
    prompt = f"""Generate a summary of the following work experience and education in 600 words:
Work experience: {work}
Education: {education}
"""
    summary_chunks = []
    try:
        async for chunk in llm.astream(prompt):
            content = chunk.content
            if content:
                summary_chunks.append(content)
                yield content
    except Exception as e:
        yield f"Error generating summary: {str(e)}"
        return

   
    full_summary = "".join(summary_chunks)
    if len(SUMMARY_CACHE) < 1000: 
        SUMMARY_CACHE[cache_key] = full_summary
    print(f"Cached summary for key: {cache_key}")

if __name__ == "__main__":
    async def test_summary_generator():
        """
        Test summary_generator with sample inputs, running twice to verify caching.
        """
        sample_work = """
        Software Engineer at TechCorp (2020-01 - Present): Developed web applications using Python and Django.
        Data Analyst at DataInc (2018-06 - 2019-12): Analyzed large datasets.
        """
        sample_education = """
        B.S. in Computer Science at State University (2014 - 2018)
        Coding Bootcamp at CodeAcademy (2017)
        """

        print("\n=== First Run (Generate New Summary) ===")
        full_summary = []
        async for chunk in summary_generator(sample_work, sample_education):
            print(f"Chunk: {chunk}")
            full_summary.append(chunk)
        print("\nFull Summary:", "".join(full_summary))

        print("\n=== Second Run (Use Cache) ===")
        full_summary = []
        async for chunk in summary_generator(sample_work, sample_education):
            print(f"Chunk: {chunk}")
            full_summary.append(chunk)
        print("\nFull Summary:", "".join(full_summary))

    asyncio.run(test_summary_generator())