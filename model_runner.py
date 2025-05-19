import os
import io
from PIL import Image
from pdf2image import convert_from_path
import pytesseract # For OCR
import google.generativeai as genai
from typing import Tuple # Keep for clarity or if using older Python versions sometimes helps linters

# --- Configuration ---

# 1. MANUALLY SET YOUR GOOGLE API KEY HERE
GOOGLE_API_KEY_MANUAL = "AIzaSyAEDhV00ymUFp2V9QpTGyDjEjgjF-nQ6_Y" # <--- REPLACE WITH YOUR ACTUAL API KEY

if GOOGLE_API_KEY_MANUAL == "YOUR_API_KEY_HERE":
    print("Error: Please replace 'YOUR_API_KEY_HERE' with your actual Google API Key in the script.")
    exit()

try:
    genai.configure(api_key=GOOGLE_API_KEY_MANUAL)
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"An error occurred during Gemini API configuration: {e}")
    exit()

# 2. MANUALLY SET THE PATH TO TESSERACT EXECUTABLE IF NEEDED
# On Windows, if Tesseract is not in your PATH, uncomment and set the line below.
# For other OS, adjust the path accordingly if tesseract is not found automatically.
# If Tesseract is in your PATH, you might be able to leave this commented out.
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # <--- UPDATE THIS PATH IF YOURS IS DIFFERENT
    # Example for Linux (if not in standard PATH):
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    # Example for macOS (if installed with Homebrew and not in PATH):
    # pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract' # For Apple Silicon
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' # For Intel Macs
    # Test if tesseract command is working
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version {version} found at: {pytesseract.pytesseract.tesseract_cmd}")
    except pytesseract.TesseractNotFoundError:
        print(f"Warning: Tesseract not found at the specified path: {pytesseract.pytesseract.tesseract_cmd}")
        print("Pytesseract might not work. Ensure Tesseract OCR is installed and the path is correct or Tesseract is in your system PATH.")
    except Exception as e:
        print(f"Note: Could not get Tesseract version (path: {pytesseract.pytesseract.tesseract_cmd}). Error: {e}")
        print("This might be okay if Tesseract is still callable by the system.")

except Exception as e:
    print(f"An error occurred setting tesseract_cmd: {e}")
    print("If you are not on Windows or Tesseract is in a different location, please update the path.")
    print("If Tesseract is in your system PATH, you might be able to comment out the `pytesseract.tesseract_cmd` line.")


# --- Agent Definitions ---

class DocumentIngestionAgent:
    def __init__(self):
        print("DocumentIngestionAgent initialized.")

    def _assess_page_for_llm_direct(self, image: Image.Image) -> bool:
        """
        Performs an initial analysis to decide if direct LLM transcription is likely successful.
        """
        if image.width < 100 or image.height < 100:
            return False

        non_white_pixels = 0
        for pixel in image.convert("L").getdata(): # Convert to grayscale
            if pixel < 200: # Assuming dark text on light background
                non_white_pixels += 1
        
        if (non_white_pixels / (image.width * image.height)) > 0.03:
            print("Page assessment: Likely suitable for direct LLM.")
            return True
        else:
            print("Page assessment: Likely requires Hybrid OCR.")
            return False

    def process_pdf(self, pdf_path: str) -> list:
        """
        Receives a PDF file, splits it into pages, assesses each page,
        and flags it for processing.
        """
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            return []

        print(f"Processing PDF: {pdf_path}")
        try:
            images_from_path = convert_from_path(pdf_path)
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            print("Ensure Poppler is installed and in your PATH, or specify poppler_path in convert_from_path.")
            return []

        processed_pages = []
        for i, image in enumerate(images_from_path):
            page_number = i + 1
            print(f"  Assessing Page {page_number}...")
            if self._assess_page_for_llm_direct(image):
                processing_type = "LLM_PROCESSING"
            else:
                processing_type = "HYBRID_PROCESSING"

            processed_pages.append({
                "image": image,
                "page_number": page_number,
                "processing_type": processing_type,
                "original_document_path": pdf_path
            })
            print(f"    Page {page_number} flagged for: {processing_type}")
        return processed_pages


class LLMTextExtractionAgent:
    def __init__(self, model_name="gemini-1.5-flash-latest"):
        self.model = genai.GenerativeModel(model_name)
        print(f"LLMTextExtractionAgent initialized with model: {model_name}")

    def _pil_to_blob(self, image: Image.Image) -> dict:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return {"mime_type": "image/png", "data": img_byte_arr}

    def extract_text_from_image(self, image: Image.Image, page_number: int) -> tuple[str, float]: # TYPE HINT CORRECTION
        print(f"  LLM Agent: Extracting text from page {page_number}...")
        image_blob = self._pil_to_blob(image)
        prompt = "Transcribe all handwritten text from this image. Preserve the original layout and line breaks as accurately as possible. If the page is blank or contains no discernible text, state that."

        try:
            response = self.model.generate_content([prompt, image_blob])
            if not response.candidates:
                print(f"    Warning: LLM for page {page_number} returned no candidates (possibly blocked).")
                return "Error: No content generated, possibly due to safety filters.", 0.0
            
            transcribed_text = response.text
            confidence = 0.85 if transcribed_text else 0.3
            print(f"    LLM Agent: Page {page_number} - Extracted text (first 50 chars): '{transcribed_text[:50].replace(os.linesep, ' ')}...'")
            return transcribed_text, confidence
        except Exception as e:
            print(f"    Error during LLM text extraction for page {page_number}: {e}")
            return f"Error during LLM extraction: {e}", 0.0


class HybridOCRAgent:
    def __init__(self, llm_model_name="gemini-1.5-flash-latest"):
        self.llm_model = genai.GenerativeModel(llm_model_name)
        print(f"HybridOCRAgent initialized with LLM model: {llm_model_name}")

    def _pil_to_blob(self, image: Image.Image) -> dict:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return {"mime_type": "image/png", "data": img_byte_arr}

    def enhance_text_with_llm(self, image: Image.Image, ocr_text: str, page_number: int) -> tuple[str, float]: # TYPE HINT CORRECTION
        print(f"  Hybrid Agent: Enhancing OCR text for page {page_number} with LLM...")
        image_blob = self._pil_to_blob(image)

        if not ocr_text.strip():
            ocr_text_prompt_part = "OCR found no text. Please transcribe the handwritten text from the image."
        else:
            ocr_text_prompt_part = f"The following text was extracted using OCR from the provided image:\n---\n{ocr_text}\n---\nPlease review this OCR output against the image. Correct any errors, focusing on accurately transcribing the handwritten content. Preserve the original layout and line breaks. If the OCR text is largely correct but needs minor fixes, apply them. If the OCR text is very poor, primarily rely on the image for transcription."

        prompt = f"{ocr_text_prompt_part}\nProvide the corrected and complete handwritten text from the image."

        try:
            response = self.llm_model.generate_content([prompt, image_blob])
            if not response.candidates:
                print(f"    Warning: LLM for hybrid page {page_number} returned no candidates (possibly blocked).")
                return f"Error: No content generated for hybrid page {page_number}, possibly due to safety filters.", 0.0
            enhanced_text = response.text
            confidence = 0.75
            print(f"    Hybrid Agent: Page {page_number} - Enhanced text (first 50 chars): '{enhanced_text[:50].replace(os.linesep, ' ')}...'")
            return enhanced_text, confidence
        except Exception as e:
            print(f"    Error during LLM enhancement for page {page_number}: {e}")
            return f"Error during LLM enhancement: {e}", 0.0

    def extract_text(self, image: Image.Image, page_number: int) -> tuple[str, float]: # TYPE HINT CORRECTION
        print(f"  Hybrid Agent: Performing OCR on page {page_number}...")
        try:
            ocr_text = pytesseract.image_to_string(image)
            print(f"    Hybrid Agent: Page {page_number} - OCR raw output (first 50 chars): '{ocr_text[:50].replace(os.linesep, ' ')}...'")
        except pytesseract.TesseractNotFoundError:
            print("    Error: Tesseract is not installed or not in your PATH, or tesseract_cmd is incorrect.")
            print("    Please install Tesseract OCR, ensure it's in PATH or set pytesseract.tesseract_cmd correctly.")
            return "Error: Tesseract not found.", 0.0
        except Exception as e:
            print(f"    Error during OCR for page {page_number}: {e}")
            return f"Error during OCR: {e}", 0.0

        return self.enhance_text_with_llm(image, ocr_text, page_number)


class AnswerAssemblyAgent:
    def __init__(self, model_name="gemini-1.5-pro-latest"):
        self.model = genai.GenerativeModel(model_name)
        print(f"AnswerAssemblyAgent initialized with model: {model_name}")

    def assemble_answers(self, page_data: list) -> str:
        if not page_data:
            return "No text extracted from any pages to assemble."

        print("AnswerAssemblyAgent: Assembling answers from extracted page texts...")
        page_data.sort(key=lambda p: p["page_number"])

        full_text_input = []
        for item in page_data:
            full_text_input.append(f"--- Start of Page {item['page_number']} ---\n{item['text']}\n--- End of Page {item['page_number']} ---")

        combined_text = "\n\n".join(full_text_input)
        prompt = (
            "You are given text extracted from sequential pages of a handwritten answer sheet. "
            "The text for each page is provided, clearly demarcated.\n"
            "Your task is to identify distinct questions and consolidate their full answers. "
            "Answers may span across multiple pages. "
            "Assume the questions are implicitly numbered or clearly indicated (e.g., 'Q1', '1.', 'Question A:').\n"
            "Structure the output as follows:\n"
            "Question 1: [Full answer for question 1, consolidated from all relevant page segments]\n"
            "Question 2: [Full answer for question 2, consolidated from all relevant page segments]\n"
            "...\n\n"
            "If the content is not clearly question-answer format, summarize the content of each logical section.\n"
            "Here is the combined text from all pages:\n\n"
            f"{combined_text}"
        )

        print("  AnswerAssemblyAgent: Sending combined text to LLM for final assembly...")
        try:
            response = self.model.generate_content(prompt)
            if not response.candidates:
                print("    Warning: Answer Assembly LLM returned no candidates (possibly blocked).")
                return "Error: No content generated by assembly model, possibly due to safety filters."
            final_answers = response.text
            print("  AnswerAssemblyAgent: Successfully assembled answers.")
            return final_answers
        except Exception as e:
            print(f"    Error during answer assembly with LLM: {e}")
            return f"Error during answer assembly: {e}"

# --- Main Workflow Orchestration ---
def main_workflow(pdf_file_path: str):
    print("Starting workflow...")

    ingestion_agent = DocumentIngestionAgent()
    llm_extraction_agent = LLMTextExtractionAgent()
    hybrid_ocr_agent = HybridOCRAgent()
    answer_assembly_agent = AnswerAssemblyAgent()

    pages_to_process = ingestion_agent.process_pdf(pdf_file_path)

    if not pages_to_process:
        print("No pages processed from PDF. Exiting.")
        return

    extracted_page_texts = []
    for page_info in pages_to_process:
        page_image = page_info["image"]
        page_number = page_info["page_number"]
        processing_type = page_info["processing_type"]
        text = ""
        confidence = 0.0

        print(f"\nProcessing Page {page_number} using {processing_type}...")

        if processing_type == "LLM_PROCESSING":
            text, confidence = llm_extraction_agent.extract_text_from_image(page_image, page_number)
        elif processing_type == "HYBRID_PROCESSING":
            text, confidence = hybrid_ocr_agent.extract_text(page_image, page_number)
        else:
            print(f"  Warning: Unknown processing type '{processing_type}' for page {page_number}. Skipping.")
            text = f"Error: Unknown processing type '{processing_type}'"
            confidence = 0.0

        extracted_page_texts.append({
            "page_number": page_number,
            "text": text,
            "confidence": confidence
        })

    print("\nAll pages processed. Collected texts:")
    for item in sorted(extracted_page_texts, key=lambda x: x['page_number']):
        print(f"  Page {item['page_number']} (Confidence: {item['confidence']:.2f}): {item['text'][:100].replace(os.linesep, ' ')}...")

    print("\nStarting Answer Assembly...")
    final_machine_readable_answers = answer_assembly_agent.assemble_answers(extracted_page_texts)

    print("\n--- FINAL MACHINE-READABLE ANSWERS ---")
    print(final_machine_readable_answers)
    print("\n--- WORKFLOW COMPLETE ---")

    output_filename = os.path.splitext(os.path.basename(pdf_file_path))[0] + "_answers.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("FINAL MACHINE-READABLE ANSWERS (from " + pdf_file_path + "):\n\n")
        f.write(final_machine_readable_answers)
    print(f"Final answers also saved to: {output_filename}")


if __name__ == "__main__":
    # 3. SET THE PDF FILE TO PROCESS HERE
    pdf_to_process = "input.pdf" # Defaulting to input.pdf in the same folder

    if GOOGLE_API_KEY_MANUAL == "YOUR_API_KEY_HERE": # Check if API key placeholder is still there
        print("CRITICAL ERROR: GOOGLE_API_KEY_MANUAL has not been set in the script.")
        print("Please edit the script and replace 'YOUR_API_KEY_HERE' with your actual API key.")
    elif not os.path.exists(pdf_to_process):
        print(f"Error: PDF file '{pdf_to_process}' not found in the current directory: {os.getcwd()}")
        print("Please make sure 'input.pdf' is in the same folder as the script, or provide the correct path.")
        # Optionally, you could ask for input here again:
        # pdf_to_process = input(f"Enter the path to your PDF answer sheet (default: {pdf_to_process}): ") or pdf_to_process
        # if not os.path.exists(pdf_to_process):
        #     print(f"Error: PDF file still not found at '{pdf_to_process}'. Exiting.")
        # else:
        #     main_workflow(pdf_to_process)
    else:
        main_workflow(pdf_to_process)