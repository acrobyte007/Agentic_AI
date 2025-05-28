import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")




llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    api_key=api_key
)




def summary_generator(Work:list,education:list):
        print("inside summary generator function")
        prompt = f"""Generate a summary of the following work experience and education:
Work experience: {Work}
Education: {education}
"""
        response = llm.invoke(prompt)
        print("Generating summary is called")
        return response.content  ##string

