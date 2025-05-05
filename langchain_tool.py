# langchain_tool.py

from langchain.tools import Tool
from ocr_preprocessing_agent import PreprocessingAgent

def run_preprocessing_tool(pdf_path: str) -> str:
    agent = PreprocessingAgent(dpi=300)
    output_files = agent.process_pdf(pdf_path)
    return f"Processed {len(output_files)} pages. Saved to: {', '.join(output_files)}"

preprocessing_tool = Tool(
    name="OCRPreprocessingTool",
    func=run_preprocessing_tool,
    description="Use this tool to preprocess scanned exam PDFs for OCR. Input is a file path to a PDF."
)
