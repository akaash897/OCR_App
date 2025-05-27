import os
import io
import glob
from PIL import Image
from pdf2image import convert_from_path
import pytesseract # For OCR
import google.generativeai as genai
from typing import Tuple, List
import time
from datetime import datetime

# --- Configuration ---

# 1. MANUALLY SET YOUR GOOGLE API KEYS HERE
GOOGLE_API_KEY_PAGE_PROCESSING_MANUAL = "NA" # <--- REPLACE
GOOGLE_API_KEY_ASSEMBLY_MANUAL = "NA" # <--- REPLACE

# Model Configuration
# Using gemini-2.0-flash as requested for all LLM tasks
DEFAULT_LLM_MODEL = "gemini-2.0-flash"

# Rate Limiting Configuration
API_CALLS_PER_MINUTE_LIMIT = 15 # Your API limit
API_CALL_DELAY_SECONDS = (60 / API_CALLS_PER_MINUTE_LIMIT) + 0.5 # Extra buffer
SAFETY_MARGIN_SECONDS = 5 # Additional safety margin every 10 calls

print(f"API call delay set to: {API_CALL_DELAY_SECONDS:.2f} seconds to respect {API_CALLS_PER_MINUTE_LIMIT} RPM limit.")
print(f"All LLM agents will use model: {DEFAULT_LLM_MODEL}")

# PDF Processing Configuration
PDF_INPUT_FOLDER = "input_pdfs"  # Folder containing PDFs to process
PDF_INPUT_PATTERN = "*.pdf"     # Pattern to match PDF files
OUTPUT_FOLDER = "output_results" # Folder for output files

# Create necessary directories
os.makedirs(PDF_INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if GOOGLE_API_KEY_PAGE_PROCESSING_MANUAL == "NA" or \
   GOOGLE_API_KEY_ASSEMBLY_MANUAL == "NA":
    print("Error: Critical - Please replace 'NA' with your actual Google API Keys in the script.")
    exit()

# 2. MANUALLY SET THE PATH TO TESSERACT EXECUTABLE IF NEEDED
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
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

# --- Rate Limiting Manager ---
class RateLimitManager:
    def __init__(self, calls_per_minute=15):
        self.calls_per_minute = calls_per_minute
        self.call_times = []
        self.total_calls = 0
        
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()
        
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if current_time - t < 60]
        
        # If we're at the limit, wait until we can make another call
        if len(self.call_times) >= self.calls_per_minute:
            sleep_time = 60 - (current_time - self.call_times[0]) + 1
            if sleep_time > 0:
                print(f"   Rate limit reached. Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
        
        # Add safety margin every 10 calls
        if self.total_calls > 0 and self.total_calls % 10 == 0:
            print(f"   Safety pause after {self.total_calls} calls. Waiting {SAFETY_MARGIN_SECONDS} seconds...")
            time.sleep(SAFETY_MARGIN_SECONDS)
        
        # Record this call
        self.call_times.append(time.time())
        self.total_calls += 1
        
        # Regular delay between calls
        time.sleep(API_CALL_DELAY_SECONDS)

# Global rate limiter
rate_limiter = RateLimitManager(API_CALLS_PER_MINUTE_LIMIT)

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

    def process_pdf(self, pdf_path: str) -> list:
        """Process PDF without page limits - handle full document"""
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
            return []

        processed_pages = []
        print(f"Total pages in PDF: {len(images_from_path)}")
        
        for i, image in enumerate(images_from_path):
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
        
        if not processed_pages:
            print("Warning: No pages were processed from the PDF. Check PDF content.")
        
        return processed_pages


class LLMTextExtractionAgent:
    def __init__(self, model_name=DEFAULT_LLM_MODEL):
        self.model = genai.GenerativeModel(model_name)
        print(f"LLMTextExtractionAgent initialized with model: {model_name}")

    def _pil_to_blob(self, image: Image.Image) -> dict:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return {"mime_type": "image/png", "data": img_byte_arr}

    def extract_text_from_image(self, image: Image.Image, page_number: int) -> tuple[str, float]:
        print(f"   LLM Agent: Extracting text from page {page_number}...")
        
        try:
            image_blob = self._pil_to_blob(image)
        except Exception as e:
            print(f"     Error converting PIL image to blob for page {page_number}: {e}")
            return f"Error preparing image blob: {e}", 0.0

        prompt = "Transcribe all handwritten text from this image. Preserve the original layout and line breaks as accurately as possible. If the page is blank or contains no discernible text, state that."
        
        try:
            # Wait for rate limiting before API call
            rate_limiter.wait_if_needed()
            
            response = self.model.generate_content([prompt, image_blob])
            
            if not response.candidates:
                error_message = f"Error: LLM for page {page_number} returned no candidates."
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
            print(f"     Error during LLM text extraction for page {page_number}: {e}")
            return f"Error during LLM extraction: {e}", 0.0


class HybridOCRAgent:
    def __init__(self, llm_model_name=DEFAULT_LLM_MODEL):
        self.llm_model = genai.GenerativeModel(llm_model_name)
        print(f"HybridOCRAgent initialized with LLM model: {llm_model_name}")

    def _pil_to_blob(self, image: Image.Image) -> dict:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return {"mime_type": "image/png", "data": img_byte_arr}

    def enhance_text_with_llm(self, image: Image.Image, ocr_text: str, page_number: int) -> tuple[str, float]:
        print(f"   Hybrid Agent: Enhancing OCR text for page {page_number}...")
        
        try:
            image_blob = self._pil_to_blob(image)
        except Exception as e:
            print(f"     Error converting PIL image to blob for hybrid page {page_number}: {e}")
            return f"Error preparing image blob for hybrid: {e}", 0.0

        if not ocr_text.strip():
            ocr_text_prompt_part = "OCR found no text or only whitespace. Please transcribe the handwritten text directly from the image."
        else:
            ocr_text_prompt_part = f"The following text was extracted using OCR from the provided image:\n---\n{ocr_text}\n---\nPlease review this OCR output against the image. Correct any errors, focusing on accurately transcribing the handwritten content. Preserve the original layout and line breaks."
        
        prompt = f"{ocr_text_prompt_part}\nProvide the corrected and complete handwritten text from the image."
        
        try:
            # Wait for rate limiting before API call
            rate_limiter.wait_if_needed()
            
            response = self.llm_model.generate_content([prompt, image_blob])
            
            if not response.candidates:
                error_message = f"Error: LLM for hybrid page {page_number} returned no candidates."
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
            print(f"     Error during LLM enhancement for page {page_number}: {e}")
            return f"Error during LLM enhancement: {e}", 0.0

    def extract_text(self, image: Image.Image, page_number: int) -> tuple[str, float]:
        print(f"   Hybrid Agent: Performing OCR on page {page_number}...")
        ocr_text = ""
        
        try:
            ocr_text = pytesseract.image_to_string(image)
            if not ocr_text.strip():
                print(f"     Hybrid Agent: Page {page_number} - OCR found no text.")
            else:
                print(f"     Hybrid Agent: Page {page_number} - OCR output (first 50 chars): '{ocr_text[:50].replace(os.linesep, ' ')}...'")
        except pytesseract.TesseractNotFoundError:
            print("     Critical Error: Tesseract not found.")
            return "Critical Error: Tesseract not found.", 0.0
        except Exception as e:
            print(f"     Error during OCR for page {page_number}: {e}")
            return f"Error during OCR: {e}", 0.0
        
        return self.enhance_text_with_llm(image, ocr_text, page_number)


class AnswerAssemblyAgent:
    def __init__(self, model_name=DEFAULT_LLM_MODEL):
        self.model = genai.GenerativeModel(model_name)
        print(f"AnswerAssemblyAgent initialized with model: {model_name}")

    def assemble_answers(self, page_data: list) -> str:
        if not page_data:
            return "No text extracted from any pages to assemble."
        
        valid_texts_count = sum(1 for item in page_data if "Error:" not in item['text'])
        if valid_texts_count == 0:
            print("AnswerAssemblyAgent: No valid text extracted. Skipping assembly.")
            return "Error: No valid text was extracted from the processed pages."

        print(f"AnswerAssemblyAgent: Assembling answers from {len(page_data)} pages...")
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
        
        try:
            # Wait for rate limiting before API call
            rate_limiter.wait_if_needed()
            
            response = self.model.generate_content(prompt)
            
            if not response.candidates:
                error_message = f"Error: Answer Assembly LLM returned no candidates."
                print(f"     {error_message}")
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    print(f"     Prompt Feedback: {response.prompt_feedback}")
                    error_message += f" Feedback: {response.prompt_feedback}"
                return error_message
                
            final_answers = response.text
            print("   AnswerAssemblyAgent: Successfully assembled answers.")
            return final_answers
            
        except Exception as e:
            print(f"     Error during answer assembly: {e}")
            return f"Error during answer assembly: {e}"


# --- Multi-PDF Processing Functions ---

def get_pdf_files() -> List[str]:
    """Get list of PDF files to process"""
    pdf_pattern = os.path.join(PDF_INPUT_FOLDER, PDF_INPUT_PATTERN)
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print(f"No PDF files found in {PDF_INPUT_FOLDER} matching pattern {PDF_INPUT_PATTERN}")
        print(f"Please place PDF files in the '{PDF_INPUT_FOLDER}' folder.")
    
    return sorted(pdf_files)


def process_single_pdf(pdf_path: str, pdf_index: int, total_pdfs: int) -> bool:
    """Process a single PDF and save results"""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"\n{'='*60}")
    print(f"PROCESSING PDF {pdf_index}/{total_pdfs}: {pdf_name}")
    print(f"{'='*60}")
    
    try:
        # Initialize agents
        ingestion_agent = DocumentIngestionAgent()
        llm_extraction_agent = LLMTextExtractionAgent()
        hybrid_ocr_agent = HybridOCRAgent()
        
        # Process PDF pages
        pages_to_process = ingestion_agent.process_pdf(pdf_path)
        
        if not pages_to_process:
            print(f"No pages processed from {pdf_name}. Skipping.")
            return False
        
        # Extract text from each page
        extracted_page_texts = []
        for page_info in pages_to_process:
            page_image = page_info["image"]
            page_number = page_info["page_number"]
            processing_type = page_info["processing_type"]
            
            print(f"\nProcessing Page {page_number}/{len(pages_to_process)} using {processing_type}...")
            
            if processing_type == "LLM_PROCESSING":
                text, confidence = llm_extraction_agent.extract_text_from_image(page_image, page_number)
            elif processing_type == "HYBRID_PROCESSING":
                text, confidence = hybrid_ocr_agent.extract_text(page_image, page_number)
            else:
                print(f"   Warning: Unknown processing type '{processing_type}' for page {page_number}.")
                text = f"Error: Unknown processing type '{processing_type}'"
                confidence = 0.0
            
            extracted_page_texts.append({
                "page_number": page_number,
                "text": text,
                "confidence": confidence,
                "processing_type": processing_type
            })
        
        # Print processing summary
        print(f"\n--- Processing Summary for {pdf_name} ---")
        for item in extracted_page_texts:
            status = "OK" if "Error:" not in item['text'] else "ERROR"
            print(f"  Page {item['page_number']} ({item['processing_type']}, {status}, Confidence: {item['confidence']:.2f})")
        
        # Switch to assembly API key
        print(f"\nSwitching to Assembly API key for {pdf_name}...")
        genai.configure(api_key=GOOGLE_API_KEY_ASSEMBLY_MANUAL)
        
        # Assemble final answers
        answer_assembly_agent = AnswerAssemblyAgent()
        final_answers = answer_assembly_agent.assemble_answers(extracted_page_texts)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(OUTPUT_FOLDER, f"{pdf_name}_results_{timestamp}.txt")
        
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"HANDWRITTEN TEXT EXTRACTION RESULTS\n")
            f.write(f"Source PDF: {pdf_path}\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Used: {DEFAULT_LLM_MODEL}\n")
            f.write(f"Total Pages Processed: {len(pages_to_process)}\n")
            f.write(f"API Calls Made: {rate_limiter.total_calls}\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"ASSEMBLED ANSWERS:\n")
            f.write(f"{'='*60}\n\n")
            f.write(final_answers)
            f.write(f"\n\n{'='*60}\n")
            f.write(f"INDIVIDUAL PAGE EXTRACTIONS:\n")
            f.write(f"{'='*60}\n\n")
            
            for item in extracted_page_texts:
                f.write(f"--- Page {item['page_number']} ({item['processing_type']}) ---\n")
                f.write(f"Confidence: {item['confidence']:.2f}\n")
                f.write(f"Text:\n{item['text']}\n\n")
        
        print(f"\nResults saved to: {output_filename}")
        
        # Switch back to page processing API key for next PDF
        genai.configure(api_key=GOOGLE_API_KEY_PAGE_PROCESSING_MANUAL)
        
        return True
        
    except Exception as e:
        print(f"Error processing {pdf_name}: {e}")
        return False


def main_multi_pdf_workflow():
    """Main workflow to process multiple PDFs"""
    print("MULTI-PDF HANDWRITTEN TEXT EXTRACTION WORKFLOW")
    print(f"Rate Limit: {API_CALLS_PER_MINUTE_LIMIT} calls per minute")
    print(f"Model: {DEFAULT_LLM_MODEL}")
    print(f"Input Folder: {PDF_INPUT_FOLDER}")
    print(f"Output Folder: {OUTPUT_FOLDER}")
    
    # Configure initial API key
    try:
        genai.configure(api_key=GOOGLE_API_KEY_PAGE_PROCESSING_MANUAL)
        print("✓ Page Processing API key configured")
    except Exception as e:
        print(f"✗ Error configuring Page Processing API key: {e}")
        return
    
    # Get PDF files to process
    pdf_files = get_pdf_files()
    if not pdf_files:
        return
    
    print(f"\nFound {len(pdf_files)} PDF file(s) to process:")
    for i, pdf_path in enumerate(pdf_files, 1):
        pdf_name = os.path.basename(pdf_path)
        print(f"  {i}. {pdf_name}")
    
    # Process each PDF
    successful_pdfs = 0
    start_time = time.time()
    
    for i, pdf_path in enumerate(pdf_files, 1):
        if process_single_pdf(pdf_path, i, len(pdf_files)):
            successful_pdfs += 1
        
        # Add extra delay between PDFs to be safe
        if i < len(pdf_files):
            print(f"\nWaiting before next PDF... (Total API calls so far: {rate_limiter.total_calls})")
            time.sleep(5)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"WORKFLOW COMPLETE")
    print(f"{'='*60}")
    print(f"Total PDFs processed: {successful_pdfs}/{len(pdf_files)}")
    print(f"Total processing time: {total_time/60:.1f} minutes")
    print(f"Total API calls made: {rate_limiter.total_calls}")
    print(f"Results saved in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    # Validate API keys
    if GOOGLE_API_KEY_PAGE_PROCESSING_MANUAL == "NA" or GOOGLE_API_KEY_ASSEMBLY_MANUAL == "NA":
        print("CRITICAL ERROR: Please set your actual Google API keys in the script.")
        print("Replace 'NA' with your API keys in GOOGLE_API_KEY_PAGE_PROCESSING_MANUAL and GOOGLE_API_KEY_ASSEMBLY_MANUAL")
    else:
        main_multi_pdf_workflow()