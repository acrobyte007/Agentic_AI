from work_exp import work_exp
from educational_exp import edu_exp
from summary import summary_generator
from insight_extractor import extract_insights
from questions_generation import generate_questions
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage, HumanMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]
    Work: Annotated[list, add_messages]
    education: Annotated[list, add_messages]
    resume_text: Annotated[list, add_messages]

def work_exp_generator(state: State):
    # Assuming work_exp returns a list of dictionaries
    work_data = work_exp(state['resume_text'][-1].content)  
    # Convert work_data (list of dicts) to a string for AIMessage
    work_str = "\n".join(
        f"{job['role']} at {job['company']} ({job['start_date']} - {job['end_date']}): {job['description']}"
        for job in work_data
    )
    return {"Work": [AIMessage(content=work_str)], "messages": [AIMessage(content=work_str)]}

def edu_exp_generator(state: State):
    # Assuming edu_exp returns a list of dictionaries or a string
    education_data = edu_exp(state['resume_text'][-1].content)  # Access content of resume_text
    # If education_data is a list of dictionaries, convert to string
    if isinstance(education_data, list):
        education_str = "\n".join(
            f"{edu.get('degree', 'Unknown Degree')} at {edu.get('institution', 'Unknown Institution')} ({edu.get('start_date', 'Unknown')} - {edu.get('end_date', 'Unknown')})"
            for edu in education_data
        )
    else:
        education_str = education_data  # Assume it's already a string
    return {"education": [AIMessage(content=education_str)], "messages": [AIMessage(content=education_str)]}

def makes_summary(state: State):
    summary = summary_generator(state['Work'][-1].content, state['education'][-1].content)  # Use content of latest entries
    print("data type of summary", type(summary))
    return {"messages": [AIMessage(content=summary)]}

def insight_extractor(state: State):
    insights = extract_insights(state['messages'][-1].content)  # Extract from latest message content
    return {"messages": [AIMessage(content=insights)]}

def questions_generator(state: State):
    questions = generate_questions(state['messages'][-1].content)  # Extract from latest message content
    return {"messages": [AIMessage(content=questions)]}

# Workflow setup
workflow = StateGraph(State)
workflow.add_node("work_exp", work_exp_generator)
workflow.add_node("edu_exp", edu_exp_generator)
workflow.add_node("summary", makes_summary)
workflow.add_node("insights", insight_extractor)
workflow.add_node("questions", questions_generator)

# Define edges (sequential flow)
workflow.set_entry_point("work_exp")
workflow.add_edge("work_exp", "edu_exp")
workflow.add_edge("edu_exp", "summary")
workflow.add_edge("summary", "insights")
workflow.add_edge("insights", "questions")

# Compile the graph
graph = workflow.compile()

initial_state = {
    "resume_text": [HumanMessage(content="""  # Ensure resume_text is a HumanMessage
    John Doe
    Work Experience:
    - Software Engineer, TechCorp, 2020-2023: Developed web applications using Python and Django.
    - Data Scientist, DataInc, 2023-2025: Built machine learning models for predictive analytics.
    Education:
    - B.S. Computer Science, University of Example, 2016-2020
    """)],
    "messages": [],
    "Work": [],
    "education": []
}

# Invoke the graph
result = graph.invoke(initial_state)
# Get the last message from the result
last_message = result['messages'][-1].content

print("Last message:", last_message)