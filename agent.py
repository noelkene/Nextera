import os
import csv
import re
from typing import Dict, List, Any
from PyPDF2 import PdfReader
from docx import Document
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
import vertexai
from agents import Agent
from agents.tools import ToolContext
import google.generativeai as genai
import random

# --- Configuration ---
PROJECT_ID = "platinum-banner-303105"
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)
MODEL = GenerativeModel("gemini-2.0-pro-exp-02-05")
METADATA_FILE = "samples/Progress_Agent/image_metadata.txt"
IMAGE_DIR = "samples/Progress_Agent/images"
OUTPUT_DIR = "samples/Progress_Agent/output"
DOC_DIR = "samples/Progress_Agent/documents"

# Ensure all directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOC_DIR, exist_ok=True)

# --- Helper Functions ---
def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts and returns text from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text.
    """
    try:
        with open(file_path, "rb") as f:
            pdf_reader = PdfReader(f)
            return "".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """
    Extracts and returns text from a DOCX file.

    Args:
        file_path (str): The path to the DOCX file.

    Returns:
        str: The extracted text.
    """
    try:
        doc = Document(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        print(f"Error extracting text from DOCX {file_path}: {e}")
        return ""

def analyze_document_with_gemini(text: str, prompt_modifier: str = "") -> str:
    """
    Analyzes the provided document text using Gemini Pro.

    Args:
        prompt_modifier (str): Additional instructions for the analysis.
        text (str): The text content to analyze.

    Returns:
        str: The AI-generated analysis.
    """
    prompt = f"""
    Analyze the following document. {prompt_modifier}
    Extract key milestones, current progress, blockers, and new updates.
    {text}

    Provide the results in a well-structured, human-readable format.
    """
    try:
        response = MODEL.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred during document analysis: {e}"

def analyze_images_with_gemini(image_metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyzes construction site images using Gemini Pro and returns structured results.

    Args:
        image_metadata_list (List[Dict[str, Any]]): Metadata including filenames and dates.

    Returns:
        List[Dict[str, Any]]: List of image analysis results.
    """
    analysis_results = []

    for image_metadata in image_metadata_list:
        image_filename = image_metadata['name']
        image_path = os.path.join(IMAGE_DIR, image_filename)  # Use global IMAGE_DIR
        try:
            if os.path.exists(image_path):
                image_part = Part.from_file(image_path, mime_type="image/png")
            else:
                print(f"Image file not found: {image_path}")
                continue
        except Exception as e:
            print(f"Error loading image from path {image_path}: {e}")
            continue

        prompt_text = (
            f"Analyze this satellite image of a construction site. "
            f"Consider the current state of progress, potential issues, and the overall timeline. "
            f"Provide specific details about what you observe in the image."
        )
        text_part = Part.from_text(prompt_text)

        contents = [image_part, text_part]

        try:
            response = MODEL.generate_content(
                contents,
                generation_config=GenerationConfig(
                    temperature=0.2,
                    top_p=0.95,
                    max_output_tokens=8024,
                ),
            )
            response_text = response.text
        except Exception as e:
            print(f"Error with Gemini client for image {image_filename}: {e}. Using simulated analysis.")
            simulated_progress = random.uniform(50, 100)
            response_text = (
                f"Simulated analysis for image {image_filename}. The site appears to be at approximately "
                f"{simulated_progress:.1f}% progress with visible structural developments."
            )

        progress_match = re.search(r"(\d+(\.\d+)?)\s*%?", response_text)
        progress_estimate = float(progress_match.group(1)) if progress_match else 0.0

        analysis_results.append({
            "image_name": image_filename,
            "description": response_text,
            "progress": progress_estimate,
            "date": image_metadata["date"]
        })

    comparison_summary = generate_comparison_summary_with_gemini(analysis_results)
    output_file = os.path.join(OUTPUT_DIR, 'image_comparison_summary.txt')
    with open(output_file, 'w') as f:
        f.write(comparison_summary)
    return analysis_results

def load_image_metadata() -> List[Dict[str, str]]:
    """
    Loads and returns image metadata from a CSV file.

    Returns:
        List[Dict[str, str]]: Parsed metadata rows.
    """
    metadata = []
    try:
        with open(METADATA_FILE, mode='r') as csv_file: #use the constant
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                metadata.append(row)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {METADATA_FILE}")
    except Exception as e:
        print(f"An error occurred while loading metadata: {e}")
    return metadata

def generate_comparison_summary_with_gemini(analysis_results: List[Dict[str, Any]]) -> str:
    """
    Generates a chronological comparison summary of construction progress between image pairs.

    Args:
        analysis_results (List[Dict[str, Any]]): The list of image analysis results.

    Returns:
        str: A summary comparing the progress between image snapshots.
    """
    if len(analysis_results) < 2:
        return "Not enough images to compare."
    analysis_results.sort(key=lambda x: x["date"])

    comparison_summary = ""
    for i in range(1, len(analysis_results)):
        current_analysis = analysis_results[i]["description"]
        previous_analysis = analysis_results[i - 1]["description"]
        current_date = analysis_results[i]["date"]
        previous_date = analysis_results[i - 1]["date"]

        comparison_prompt = (
            f"""
            Summarize the progress shown by comparing these two construction site analyses:

            New Analysis (Date: {current_date}): "{current_analysis}"

            Previous Analysis (Date: {previous_date}): "{previous_analysis}"

            Focus on any changes, improvements, or issues that are now present.
            Be concise and only present the main points.
            What percentage of change has occurred since the previous image?
            """
        )

        try:
            response = MODEL.generate_content([Part.from_text(comparison_prompt)])
            comparison_text = response.text
        except Exception as e:
            comparison_text = f"Error generating comparison: {e}"

        comparison_summary += f"### Progress from {previous_date} to {current_date}\n{comparison_text}\n\n"

    return comparison_summary

# --- Tool Functions ---
def contractor_update_analysis_tool(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Analyzes a document (PDF or DOCX) from the predefined document directory and extracts progress-related details.

    Args:
        tool_context (ToolContext): Execution context.

    Returns:
        Dict[str, Any]: Analysis results.
    """
    file_name = "2_contractor_update.pdf"
    file_path = os.path.join(DOC_DIR, file_name)
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    prompt_modifier = "You are a contractor update analyst. your goal is to compare the the initial project to the status as reported by the contractor. Highlight any delays or missed deadliners or cost overruns."
    analysis_result = analyze_document_with_gemini(text, prompt_modifier)
    return {"analysis_result": analysis_result}

def project_plan_analysis_tool(tool_context: ToolContext) -> Dict[str, Any]:
    """
        Analyzes a document (PDF or DOCX) from the predefined document directory and extracts progress-related details.

        Args:
            tool_context (ToolContext): Execution context.


        Returns:
            Dict[str, Any]: Analysis results.
    """
    file_name = "1_project_plan.pdf"
    file_path = os.path.join(DOC_DIR, file_name)
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)

    prompt_modifier = "You are a project plan analyst. your goal is to compare take the key insights from this plan of the project. You should see this as a plan not as a report o finished project."
    analysis_result = analyze_document_with_gemini(text, prompt_modifier)
    return {"analysis_result": analysis_result}

def image_analysis_tool(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Uses image metadata and folder structure to analyze and compare construction progress.

    Args:
        tool_context (ToolContext): Execution context.

    Returns:
        Dict[str, Any]: List of descriptions by image and comparison summary.
    """
    image_metadata = load_image_metadata()
    analysis_results = analyze_images_with_gemini(image_metadata)

    results_output = [
        {"date": result["date"], "description": result["description"]}
        for result in analysis_results
    ]
    return {"analysis_results": results_output}

# --- Agent Definitions ---
project_agent = Agent(
    model="gemini-2.0-flash-001",
    name="project_agent",
    description="Analyzes project-related files (PDF or DOCX) in the document folder.",
    tools=[project_plan_analysis_tool],
    instruction="""
    You are the project_agent. Your role is to process project-related files (PDF or DOCX) from the document folder.
    - From each file, extract key milestones, current progress, blockers, and new updates.
    - Compare findings to known project plans and flag any potential delays or risks.
    Use the document_analysis_tool to produce a concise, human-readable summary.
    Only process documents that appear to be project plans or reports.
    """,
)

contractor_agent = Agent(
    model="gemini-2.0-flash-001",
    name="contractor_agent",
    description="Analyzes contractor-related files (PDF or DOCX) in the document folder.",
    tools=[contractor_update_analysis_tool],
    instruction="""
    You are the contractor_agent. Your role is to process contractor-related files (PDF or DOCX) from the document folder (e.g., subcontractor updates, labor reports).
    - Extract milestones, updates, risks, and blockers.
    - Identify any misalignment with the overall project plan.
    Use the document_analysis_tool to generate a human-readable summary.
    Only process documents that appear to be from contractors or subcontractors.
    """,
)

image_agent = Agent(
    model="gemini-2.0-flash-001",
    name="image_agent",
    description="Analyzes construction site images in the image folder using the corresponding metadata file.",
    tools=[image_analysis_tool],
    instruction="""
    You are the image_agent. Your role is to analyze images from the image directory using the metadata file.
    - Load metadata for each image.
    - Analyze construction progress in each image using the image_analysis_tool.
    - Compare changes across time using metadata dates.
    - Store a written comparison summary in the output folder.
    Only analyze images listed in the metadata file.
    """,
)

root_agent = Agent(
    model="gemini-2.0-flash-001",
    name="root_agent",
    description="Coordinates sub-agents to process files in the document and image folders.",
    flow="sequential",
    children=[project_agent, contractor_agent, image_agent],
    instruction="""
    You are the root_agent. Your job is to coordinate the analysis of documents and images in predefined folders.
    - Retrieve the 1_project_plan.pdf from DOC_DIR, send to the project_agent.
    - Retrieve the 2_contractor_update.pdf from DOC_DIR, send to the contractor-related.
    - For images and the metadata file in IMAGE_DIR and METADATA_FILE, trigger the image_agent.
    Assume all files are already downloaded and do not wait for uploads.
    """,
)
