import os
import re
import json
import pandas as pd
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader
from textwrap import dedent
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Crew, Process, Agent, Task

# Ensure NLTK data is downloaded
import nltk
nltk.download('punkt')

# Google LLM setup
llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            verbose=True,
            temperature=0.5,
            google_api_key=''
        )

def get_pdf_text_by_page(pdf_path):
    """Extract text from each page of a PDF file."""
    pdf_reader = PdfReader(pdf_path)
    pages_text = []
    for i, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text() if page.extract_text() else ""
        pages_text.append(page_text)
    return pages_text

def clean_text(text):
    """Clean the extracted text using regular expressions."""
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove any non-printable characters (optional)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text

def process_text(text):
    """Process the cleaned text into sentences using NLTK."""
    sentences = sent_tokenize(text)
    return sentences

def save_text_to_file(pages, output_path):
    """Save the extracted text to a .txt file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        for i, page_text in enumerate(pages):
            separator = f"\n{'-'*20}\nPage {i + 1}\n{'-'*20}\n"
            file.write(separator + page_text + '\n')

def save_qa_to_csv(qa_list, output_path):
    """Save the generated Q&A pairs to a CSV file."""
    df = pd.DataFrame(qa_list, columns=["Page", "Question", "Answer"])
    df.to_csv(output_path, index=False)

def extract_questions_answers(json_data):
    questions = []
    answers = []
    
    # Print the JSON data to understand its structure
    print("JSON Data:", json_data)
    
    if isinstance(json_data, list):
        for qa_pair in json_data:
            if "question" in qa_pair and "answer" in qa_pair:
                questions.append(qa_pair["question"])
                answers.append(qa_pair["answer"])
    elif isinstance(json_data, dict) and "questions" in json_data:
        for qa_pair in json_data["questions"]:
            if "question" in qa_pair and "answer" in qa_pair:
                questions.append(qa_pair["question"])
                answers.append(qa_pair["answer"])
    else:
        print("Unexpected JSON structure.")
    
    return questions, answers

def generate_pages():
    pdf_path = 'data/Data.pdf'  # Relative path to your PDF file
    text_output_path = 'QA/QA_data.txt'  # Relative path to your output text file
    qa_output_path = 'QA/QA_data.csv'  # Relative path to your output CSV file

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(text_output_path), exist_ok=True)

    # Extract text from the PDF
    pages_text = get_pdf_text_by_page(pdf_path)

    # Process each page
    processed_pages = []
    qa_list = []
    for i, page_text in enumerate(pages_text):
        cleaned_text = clean_text(page_text)
        sentences = process_text(cleaned_text)
        page_output = ' '.join(sentences)
        processed_pages.append(page_output)
        
        questions_answers = generate_QA(page_text)
        
        # Print the raw JSON string to debug
        print("Raw JSON String:", questions_answers)
        
        # Clean up JSON string if necessary
        questions_answers = questions_answers.replace('\n', ' ').replace('\r', ' ')
        
        try:
            json_data = json.loads(questions_answers)
            questions, answers = extract_questions_answers(json_data)
            for j, (question, answer) in enumerate(zip(questions, answers), start=1):
                qa_list.append([i + 1, question, answer])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            continue

    # Save the processed text to a .txt file
    save_text_to_file(processed_pages, text_output_path)
    # Save the Q&A pairs to a .csv file
    save_qa_to_csv(qa_list, qa_output_path)

    print(f"Text extracted and saved to {text_output_path}")
    print(f"Q&A pairs generated and saved to {qa_output_path}")

def QA_agent():
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
    return Task(
        description=f"""Generate questions and detailed answers based on the provided context\n\n {page_text}.""",
        expected_output="""List of dictionaries containing questions and their corresponding answers.""",
        context=[],
        input_data=page_text,
        agent=agent,
    )

def generate_QA(page_text):
    agent = QA_agent()
    task = draft_QA_task(agent, page_text)
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        process=Process.sequential,
        full_output=True,
        share_crew=False,
    )
    results = crew.kickoff()
    return results['final_output']

generate_pages()
