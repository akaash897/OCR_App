import os
import io
from PIL import Image
from pdf2image import convert_from_path
import pytesseract # For OCR
import google.generativeai as genai
from typing import Tuple
import time

# --- Configuration ---

# 1. MANUALLY SET YOUR GOOGLE API KEYS HERE
GOOGLE_API_KEY_PAGE_PROCESSING_MANUAL = "NA" # <--- REPLACE
GOOGLE_API_KEY_ASSEMBLY_MANUAL = "NA" # <--- REPLACE

# Model Configuration
# Using gemini-2.0-flash as requested for all LLM tasks
DEFAULT_LLM_MODEL = "gemini-2.0-flash" # <--- USING GEMINI 2.0 FLASH

# Rate Limiting Configuration
API_CALLS_PER_MINUTE_LIMIT = 15 # Assuming this is a general limit per key
API_CALL_DELAY_SECONDS = (60 / API_CALLS_PER_MINUTE_LIMIT) + 1 # Approx 5 seconds
print(f"API call delay set to: {API_CALL_DELAY_SECONDS:.2f} seconds to respect {API_CALLS_PER_MINUTE_LIMIT} RPM limit (per key).")
print(f"All LLM agents will use model: {DEFAULT_LLM_MODEL}")

MAX_PAGES_TO_PROCESS_FOR_DEBUGGING = 10

if GOOGLE_API_KEY_PAGE_PROCESSING_MANUAL == "YOUR_PAGE_PROCESSING_API_KEY_HERE" or \
   GOOGLE_API_KEY_ASSEMBLY_MANUAL == "YOUR_ASSEMBLY_API_KEY_HERE":
    print("Error: Critical - Please replace 'YOUR_PAGE_PROCESSING_API_KEY_HERE' and/or 'YOUR_ASSEMBLY_API_KEY_HERE' with your actual Google API Keys in the script.")
    exit()

# 2. MANUALLY SET THE PATH TO TESSERACT EXECUTABLE IF NEEDED
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # <--- UPDATE THIS PATH IF YOURS IS DIFFERENT
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
    print("If Tesseract is in your system PATH, you might be able to comment out the `pytesseract.pytesseract_cmd` line.")

# --- Agent Definitions ---

class DocumentIngestionAgent:
    def __init__(self):
        print("DocumentIngestionAgent initialized.")

    def _assess_page_for_llm_direct(self, image: Image.Image) -> bool:
        if image.width < 100 or image.height < 100:
            print("Page assessment: Dimensions too small.")
            return False
        non_white_pixels = 0
        try:
            grayscale_image = image.convert("L")
            for pixel in grayscale_image.getdata():
                if pixel < 200:
                    non_white_pixels += 1
            density = (non_white_pixels / (image.width * image.height))
            print(f"Page assessment: Non-white pixel density: {density:.4f}")
            if density > 0.03:
                print("Page assessment: Likely suitable for direct LLM.")
                return True
            else:
                print("Page assessment: Likely requires Hybrid OCR (low density).")
                return False
        except Exception as e:
            print(f"Error during page assessment _assess_page_for_llm_direct: {e}")
            return False

    def process_pdf(self, pdf_path: str, max_pages_to_process: int = None) -> list:
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            return []
        print(f"Processing PDF: {pdf_path}")
        try:
            images_from_path = convert_from_path(pdf_path)
            if not images_from_path:
                print("Error: convert_from_path returned no images. Check PDF and Poppler installation.")
                return []
        except Exception as e:
            print(f"Error converting PDF to images (convert_from_path): {e}")
            print("Ensure Poppler is installed and in your PATH, or specify poppler_path in convert_from_path.")
            print("Common issues: Poppler not found, PDF is corrupted or password-protected.")
            return []

        processed_pages = []
        pages_processed_count = 0
        print(f"Total pages in PDF: {len(images_from_path)}")
        if max_pages_to_process is not None:
            print(f"Will process a maximum of {max_pages_to_process} pages.")

        for i, image in enumerate(images_from_path):
            if max_pages_to_process is not None and pages_processed_count >= max_pages_to_process:
                print(f"Reached maximum limit of {max_pages_to_process} pages to process. Stopping PDF ingestion for further pages.")
                break
            
            page_number = i + 1
            print(f"   Assessing Page {page_number} of {len(images_from_path)}...")
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
            print(f"     Page {page_number} flagged for: {processing_type}")
            pages_processed_count += 1
        
        if not processed_pages:
            print("Warning: No pages were processed from the PDF. Check PDF content and processing limits.")
        return processed_pages


class LLMTextExtractionAgent:
    def __init__(self, model_name=DEFAULT_LLM_MODEL): # Using DEFAULT_LLM_MODEL
        self.model = genai.GenerativeModel(model_name)
        print(f"LLMTextExtractionAgent initialized with model: {model_name} (using current global API key).")

    def _pil_to_blob(self, image: Image.Image) -> dict:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return {"mime_type": "image/png", "data": img_byte_arr}

    def extract_text_from_image(self, image: Image.Image, page_number: int) -> tuple[str, float]:
        print(f"   LLM Agent: Preparing to extract text from page {page_number} using {self.model.model_name}...")
        try:
            image_blob = self._pil_to_blob(image)
        except Exception as e:
            print(f"     Error converting PIL image to blob for page {page_number}: {e}")
            return f"Error preparing image blob: {e}", 0.0

        prompt = "Transcribe all handwritten text from this image. Preserve the original layout and line breaks as accurately as possible. If the page is blank or contains no discernible text, state that."
        print(f"   LLM Agent: Sending API request for page {page_number}...")
        try:
            response = self.model.generate_content([prompt, image_blob])
            print(f"   LLM Agent: API call for page {page_number} complete. Waiting for {API_CALL_DELAY_SECONDS:.2f}s...")
            time.sleep(API_CALL_DELAY_SECONDS)
            
            if not response.candidates:
                error_message = f"Error: LLM for page {page_number} (model {self.model.model_name}) returned no candidates."
                print(f"     {error_message}")
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    print(f"     Prompt Feedback: {response.prompt_feedback}")
                    error_message += f" Feedback: {response.prompt_feedback}"
                return error_message, 0.0
            
            transcribed_text = response.text
            confidence = 0.85 if transcribed_text and transcribed_text.strip() else 0.3
            print(f"     LLM Agent: Page {page_number} - Extracted text (first 50 chars): '{transcribed_text[:50].replace(os.linesep, ' ')}...'")
            return transcribed_text, confidence
        except Exception as e:
            print(f"     Major Error during LLM text extraction API call for page {page_number} (model {self.model.model_name}): {e}")
            return f"Error during LLM extraction API call: {e}", 0.0


class HybridOCRAgent:
    def __init__(self, llm_model_name=DEFAULT_LLM_MODEL): # Using DEFAULT_LLM_MODEL
        self.llm_model = genai.GenerativeModel(llm_model_name)
        print(f"HybridOCRAgent initialized with LLM model: {llm_model_name} (using current global API key).")

    def _pil_to_blob(self, image: Image.Image) -> dict:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return {"mime_type": "image/png", "data": img_byte_arr}

    def enhance_text_with_llm(self, image: Image.Image, ocr_text: str, page_number: int) -> tuple[str, float]:
        print(f"   Hybrid Agent: Preparing to enhance OCR text for page {page_number} with LLM ({self.llm_model.model_name})...")
        try:
            image_blob = self._pil_to_blob(image)
        except Exception as e:
            print(f"     Error converting PIL image to blob for hybrid page {page_number}: {e}")
            return f"Error preparing image blob for hybrid: {e}", 0.0

        if not ocr_text.strip():
            ocr_text_prompt_part = "OCR found no text or only whitespace. Please transcribe the handwritten text directly from the image."
        else:
            ocr_text_prompt_part = f"The following text was extracted using OCR from the provided image:\n---\n{ocr_text}\n---\nPlease review this OCR output against the image. Correct any errors, focusing on accurately transcribing the handwritten content. Preserve the original layout and line breaks. If the OCR text is largely correct but needs minor fixes, apply them. If the OCR text is very poor, primarily rely on the image for transcription."
        prompt = f"{ocr_text_prompt_part}\nProvide the corrected and complete handwritten text from the image."
        print(f"   Hybrid Agent: Sending API request for enhancement of page {page_number}...")
        try:
            response = self.llm_model.generate_content([prompt, image_blob])
            print(f"   Hybrid Agent: API call for page {page_number} enhancement complete. Waiting for {API_CALL_DELAY_SECONDS:.2f}s...")
            time.sleep(API_CALL_DELAY_SECONDS)
            if not response.candidates:
                error_message = f"Error: LLM for hybrid page {page_number} (model {self.llm_model.model_name}) returned no candidates."
                print(f"     {error_message}")
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    print(f"     Prompt Feedback: {response.prompt_feedback}")
                    error_message += f" Feedback: {response.prompt_feedback}"
                return error_message, 0.0
            enhanced_text = response.text
            confidence = 0.75
            print(f"     Hybrid Agent: Page {page_number} - Enhanced text (first 50 chars): '{enhanced_text[:50].replace(os.linesep, ' ')}...'")
            return enhanced_text, confidence
        except Exception as e:
            print(f"     Major Error during LLM enhancement API call for page {page_number} (model {self.llm_model.model_name}): {e}")
            return f"Error during LLM enhancement API call: {e}", 0.0

    def extract_text(self, image: Image.Image, page_number: int) -> tuple[str, float]:
        print(f"   Hybrid Agent: Performing OCR on page {page_number}...")
        ocr_text = ""
        try:
            ocr_text = pytesseract.image_to_string(image)
            if not ocr_text.strip():
                 print(f"     Hybrid Agent: Page {page_number} - OCR raw output: No text found or only whitespace.")
            else:
                print(f"     Hybrid Agent: Page {page_number} - OCR raw output (first 50 chars): '{ocr_text[:50].replace(os.linesep, ' ')}...'")
        except pytesseract.TesseractNotFoundError:
            print("     Critical Error: Tesseract is not installed or not in your PATH, or tesseract_cmd is incorrect.")
            print("     Please install Tesseract OCR, ensure it's in PATH or set pytesseract.tesseract_cmd correctly.")
            return "Critical Error: Tesseract not found.", 0.0
        except Exception as e:
            print(f"     Error during OCR for page {page_number}: {e}")
            # Decide if you want to return or try to enhance. Current: returns error, LLM enhancement won't run.
            # If you want to try LLM even if OCR fails:
            # ocr_text = f"OCR Error: {e}" # Pass the error as OCR text
            return f"Error during OCR: {e}", 0.0
        
        return self.enhance_text_with_llm(image, ocr_text, page_number)


class AnswerAssemblyAgent:
    def __init__(self, model_name=DEFAULT_LLM_MODEL): # Using DEFAULT_LLM_MODEL
        self.model = genai.GenerativeModel(model_name)
        print(f"AnswerAssemblyAgent initialized with model: {model_name} (using current global API key).")

    def assemble_answers(self, page_data: list) -> str:
        if not page_data:
            return "No text extracted from any pages to assemble."
        
        valid_texts_count = sum(1 for item in page_data if "Error:" not in item['text'])
        if valid_texts_count == 0:
            print("AnswerAssemblyAgent: No valid text extracted from any pages. Skipping assembly.")
            return "Error: No valid text was extracted from the processed pages to assemble."

        print(f"AnswerAssemblyAgent: Assembling answers from extracted page texts using {self.model.model_name}...")
        page_data.sort(key=lambda p: p["page_number"])
        full_text_input = []
        for item in page_data:
            full_text_input.append(f"--- Start of Page {item['page_number']} ---\n{item['text']}\n--- End of Page {item['page_number']} ---")
        combined_text = "\n\n".join(full_text_input)
        
        prompt = (
            "You are given text extracted from sequential pages of a handwritten answer sheet. "
            "The text for each page is provided, clearly demarcated. Some pages might contain error messages if transcription failed.\n"
            "Your task is to identify distinct questions and consolidate their full answers. "
            "Answers may span across multiple pages. "
            "Assume the questions are implicitly numbered or clearly indicated (e.g., 'Q1', '1.', 'Question A:').\n"
            "Structure the output as follows:\n"
            "Question 1: [Full answer for question 1, consolidated from all relevant page segments]\n"
            "Question 2: [Full answer for question 2, consolidated from all relevant page segments]\n"
            "...\n\n"
            "If a page's text starts with 'Error:', acknowledge that transcription for that page failed and try to work with the available information from other pages.\n"
            "If the content is not clearly question-answer format, summarize the content of each logical section.\n"
            "Here is the combined text from all pages:\n\n"
            f"{combined_text}"
        )
        print(f"   AnswerAssemblyAgent: Sending combined text to LLM ({self.model.model_name}) for final assembly...")
        try:
            response = self.model.generate_content(prompt)
            print(f"   AnswerAssemblyAgent: API call for assembly complete. Waiting for {API_CALL_DELAY_SECONDS:.2f}s...")
            time.sleep(API_CALL_DELAY_SECONDS)
            if not response.candidates:
                error_message = f"Error: Answer Assembly LLM (model {self.model.model_name}) returned no candidates."
                print(f"     {error_message}")
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    print(f"     Prompt Feedback: {response.prompt_feedback}")
                    error_message += f" Feedback: {response.prompt_feedback}"
                return error_message
            final_answers = response.text
            print("   AnswerAssemblyAgent: Successfully assembled answers.")
            return final_answers
        except Exception as e:
            print(f"     Major Error during answer assembly with LLM (model {self.model.model_name}): {e}")
            return f"Error during answer assembly API call: {e}"

# --- Main Workflow Orchestration ---
def main_workflow(pdf_file_path: str):
    print("Starting workflow...")
    print(f"Attempting to process a maximum of {MAX_PAGES_TO_PROCESS_FOR_DEBUGGING} pages from the PDF.")
    print(f"All LLM operations will use the model: {DEFAULT_LLM_MODEL}")

    ingestion_agent = DocumentIngestionAgent()

    print(f"\nConfiguring Gemini API for Page Processing with key ending: ...{GOOGLE_API_KEY_PAGE_PROCESSING_MANUAL[-4:] if len(GOOGLE_API_KEY_PAGE_PROCESSING_MANUAL) > 4 else GOOGLE_API_KEY_PAGE_PROCESSING_MANUAL}")
    try:
        genai.configure(api_key=GOOGLE_API_KEY_PAGE_PROCESSING_MANUAL)
        print(f"Gemini API configured successfully for Page Processing (using model: {DEFAULT_LLM_MODEL}).")
    except Exception as e:
        print(f"CRITICAL ERROR during Gemini API configuration for Page Processing: {e}")
        print("Workflow cannot continue without successful API configuration for page processing.")
        return

    llm_extraction_agent = LLMTextExtractionAgent() # Will use DEFAULT_LLM_MODEL
    hybrid_ocr_agent = HybridOCRAgent() # Will use DEFAULT_LLM_MODEL

    pages_to_process = ingestion_agent.process_pdf(pdf_file_path, max_pages_to_process=MAX_PAGES_TO_PROCESS_FOR_DEBUGGING)

    if not pages_to_process:
        print("No pages were selected for processing from PDF. Exiting.")
        return

    extracted_page_texts = []
    for page_info in pages_to_process:
        page_image = page_info["image"]
        page_number = page_info["page_number"]
        processing_type = page_info["processing_type"]
        text = ""
        confidence = 0.0
        print(f"\nProcessing Page {page_number} (Actual Page {page_info['page_number']} from PDF) using {processing_type}...")
        if processing_type == "LLM_PROCESSING":
            text, confidence = llm_extraction_agent.extract_text_from_image(page_image, page_number)
        elif processing_type == "HYBRID_PROCESSING":
            text, confidence = hybrid_ocr_agent.extract_text(page_image, page_number)
        else:
            print(f"   Warning: Unknown processing type '{processing_type}' for page {page_number}. Skipping.")
            text = f"Error: Unknown processing type '{processing_type}'"
            confidence = 0.0
        extracted_page_texts.append({
            "page_number": page_number,
            "original_page_number": page_info['page_number'],
            "text": text,
            "confidence": confidence,
            "processing_type": processing_type
        })

    print("\n--- Individual Page Processing Summary ---")
    if not extracted_page_texts:
        print("No text was extracted from any pages.")
    for item in sorted(extracted_page_texts, key=lambda x: x['page_number']):
        status = "OK" if "Error:" not in item['text'] else "ERROR"
        print(f"  Page {item['page_number']} (Original PDF Page: {item['original_page_number']}, Type: {item['processing_type']}, Status: {status}, Confidence: {item['confidence']:.2f}): {item['text'][:100].replace(os.linesep, ' ')}...")

    print(f"\nConfiguring Gemini API for Answer Assembly with key ending: ...{GOOGLE_API_KEY_ASSEMBLY_MANUAL[-4:] if len(GOOGLE_API_KEY_ASSEMBLY_MANUAL) > 4 else GOOGLE_API_KEY_ASSEMBLY_MANUAL}")
    try:
        genai.configure(api_key=GOOGLE_API_KEY_ASSEMBLY_MANUAL)
        print(f"Gemini API configured successfully for Answer Assembly (using model: {DEFAULT_LLM_MODEL}).")
    except Exception as e:
        print(f"CRITICAL ERROR during Gemini API configuration for Answer Assembly: {e}")
        print("Cannot proceed with Answer Assembly due to API key configuration error.")
        final_machine_readable_answers = "Assembly skipped due to API configuration error. Partial results might be in logs."
        if extracted_page_texts:
             final_machine_readable_answers += "\n\n--- Partially Extracted Texts ---\n"
             for item in extracted_page_texts:
                 final_machine_readable_answers += f"\nPage {item['page_number']} (Original: {item['original_page_number']}):\n{item['text']}\n---"
    else:
        answer_assembly_agent = AnswerAssemblyAgent() # Will use DEFAULT_LLM_MODEL
        print("\nStarting Answer Assembly...")
        final_machine_readable_answers = answer_assembly_agent.assemble_answers(extracted_page_texts)

    print("\n--- FINAL MACHINE-READABLE ANSWERS (from processed pages) ---")
    print(final_machine_readable_answers)
    print("\n--- WORKFLOW COMPLETE ---")

    output_filename = os.path.splitext(os.path.basename(pdf_file_path))[0] + f"_first_{MAX_PAGES_TO_PROCESS_FOR_DEBUGGING}_pages_answers.txt"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"FINAL MACHINE-READABLE ANSWERS (using model {DEFAULT_LLM_MODEL}, from first {MAX_PAGES_TO_PROCESS_FOR_DEBUGGING} pages of " + pdf_file_path + "):\n\n")
            f.write(final_machine_readable_answers)
        print(f"Final answers also saved to: {output_filename}")
    except Exception as e:
        print(f"Error writing output file {output_filename}: {e}")

if __name__ == "__main__":
    pdf_to_process = "input.pdf" 

    if "YOUR_PAGE_PROCESSING_API_KEY_HERE" in GOOGLE_API_KEY_PAGE_PROCESSING_MANUAL or \
       "YOUR_ASSEMBLY_API_KEY_HERE" in GOOGLE_API_KEY_ASSEMBLY_MANUAL:
        print("CRITICAL ERROR: One or both GOOGLE_API_KEY variables still contain placeholder values.")
        print("Please edit the script and replace the placeholder API key values with your actual keys.")
    elif not os.path.exists(pdf_to_process):
        print(f"Error: PDF file '{pdf_to_process}' not found in the current directory: {os.getcwd()}")
        print(f"Please make sure '{os.path.basename(pdf_to_process)}' is in the same folder as the script, or provide the correct path.")
    else:
        main_workflow(pdf_to_process)