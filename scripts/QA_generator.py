from textwrap import dedent
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Crew, Process, Agent, Task


# Google LLM setup
llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            verbose=True,
            temperature=0.5,
            google_api_key=''
        )

def QA_agent():
    """
    Creates an agent for generating questions and detailed answers.

    Returns:
        Agent: A configured agent for question answering.
    """
    return Agent(
        role='Question Answer Generator - Senior Contextual Analyser and QA Generator',
        goal="""Generate questions and detailed answers in a maximum of three points with the provided context.""",
        backstory=dedent("""\
            You are an advanced AI designed to analyze contexts and generate high-quality questions and detailed answers. 
            Your goal is to understand the given context thoroughly and produce relevant questions and answers effectively 
            and efficiently."""),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
        memory=True
    )

def draft_QA_task(agent, page_text):
    """
    Drafts a task for the agent to generate questions and answers based on provided page text.

    Args:
        agent (Agent): The agent responsible for generating questions and answers.
        page_text (str): The text from which questions and answers are to be generated.

    Returns:
        Task: A configured task for question answering.
    """
    return Task(
        description=f"""Generate questions and detailed answers based on the provided context\n\n {page_text}.""",
        expected_output="""Dictionary containing questions and their corresponding answers.""",
        context=[],
        input_data=page_text,
        agent=agent,
    )

def generate_QA(page_text):
    Agent = QA_agent()
    task = draft_QA_task(Agent, page_text)

    # Create a Crew object
    crew = Crew(
        agents=[Agent],
        tasks=[task],
        verbose=True,
        process=Process.sequential,
        full_output=True,
        share_crew=False,
    )

    # Kick off the crew's work
    results = crew.kickoff()
    return results['final_output']

