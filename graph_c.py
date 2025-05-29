from work_exp import work_experience
from educational_exp import edu_exp
from summary import summary_generator
from insight_extractor import extract_insights
from questions_generation import generate_questions
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,END
from langchain_core.messages import AIMessage, HumanMessage
import re
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    messages: Annotated[list, add_messages]
    Work: Annotated[list, add_messages]
    education: Annotated[list, add_messages]
    resume_text: Annotated[list, add_messages]

def work_exp_generator(state: State):
    work_data = work_experience(state['resume_text'][-1].content)
    work_str = "\n".join(
        f"{job['role']} at {job['company']} ({job['start_date']} - {job['end_date']}): {job['description']}"
        for job in work_data
    )
    # Print node output
    print(f"\n[work_exp_generator] Output:")
    print(f"Work: {work_str}")
    print(f"Messages: {work_str}")
    return {"Work": [AIMessage(content=work_str)], "messages": [AIMessage(content=work_str)]}

def edu_exp_generator(state: State):
    resume_text = state['resume_text'][-1].content
    # Clean resume text to remove comments and normalize whitespace
    cleaned_text = "\n".join(line for line in resume_text.splitlines() if not line.strip().startswith('#'))
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text.strip())
    
    education_data = edu_exp(cleaned_text)
    print(f"[edu_exp_generator] Education data type: {type(education_data)}")
    print(f"[edu_exp_generator] Raw education data: {education_data}")
    
    if isinstance(education_data, list):
        education_str = "\n".join(
            f"{edu.get('Degree', 'Unknown Degree')} in {edu.get('Field', 'Unknown Field')} "
            f"at {edu.get('Institution', 'Unknown Institution')} "
            f"({edu.get('Start_year', 'Unknown')} - {edu.get('End_year', 'Unknown')})"
            for edu in education_data
            if edu.get('Degree') and edu.get('Institution') and edu.get('Field')  # Ensure valid entries
        ) or "No valid education entries found"
    else:
        education_str = str(education_data) if education_data else "No education data extracted"
    
    # Print node output
    print(f"\n[edu_exp_generator] Output:")
    print(f"Education: {education_str}")
    print(f"Messages: {education_str}")
    return {"education": [AIMessage(content=education_str)], "messages": [AIMessage(content=education_str)]}

def makes_summary(state: State):
    summary = summary_generator(state['Work'][-1].content, state['education'][-1].content)
    print(f"[makes_summary] Summary data type: {type(summary)}")
    # Print node output
    print(f"\n[makes_summary] Output:")
    print(f"Messages: {summary}")
    return {"messages": [AIMessage(content=summary)]}

def insight_extractor(state: State):
    insights = extract_insights(state['messages'][-1].content)
    # Print node output
    print(f"\n[insight_extractor] Output:")
    print(f"Messages: {insights}")
    return {"messages": [AIMessage(content=insights)]}

def questions_generator(state: State):
    questions = generate_questions(state['messages'][-1].content)
    if isinstance(questions, list):
        questions = "\n".join(questions)
    # Print node output
    print(f"\n[questions_generator] Output:")
    print(f"Messages: {questions}")
    return {"messages": [AIMessage(content=questions)]}

# Workflow setup
workflow = StateGraph(State)
workflow.add_node("work_exp", work_exp_generator)
workflow.add_node("edu_exp", edu_exp_generator)
workflow.add_node("summary", makes_summary)
workflow.add_node("insights", insight_extractor)
workflow.add_node("questions", questions_generator)

# Define edges
workflow.set_entry_point("work_exp")
workflow.add_edge("work_exp", "edu_exp")
workflow.add_edge("edu_exp", "summary")
workflow.add_edge("summary", "insights")
workflow.add_edge("insights", "questions")
# Compile the graph
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

if __name__ == "__main__":
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
    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke(initial_state,config)
    # Get the last message from the result
    last_message = result['messages'][-1].content

    print("\nFinal Output:")
    print("Last message:", last_message)