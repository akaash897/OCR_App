# Handwritten Answer Sheet Extractor AI

This Python script uses a multi-agent system powered by Google's Gemini AI and Tesseract OCR to extract handwritten answers from PDF documents, process them, and assemble them into a machine-readable format.

## Features

* **PDF Processing**: Converts PDF pages into images for analysis.
* **Intelligent Page Assessment**: Determines whether to use direct LLM transcription or a hybrid OCR+LLM approach for each page.
* **LLM-Powered Text Extraction**: Utilizes Google Gemini (specifically `gemini-1.5-flash-latest` by default for page extraction) to transcribe text directly from images.
* **Hybrid OCR Enhancement**: For pages less suitable for direct LLM, it uses Tesseract OCR to get an initial text extraction, which is then refined and corrected by a Gemini LLM.
* **Answer Consolidation**: A final Gemini model (`gemini-1.5-pro-latest` by default) assembles the extracted text from all pages, identifying distinct questions and consolidating their answers, even if they span multiple pages.
* **Output**: Saves the final assembled answers to a `.txt` file.

## Requirements

* Python 3.8+
* **Google API Key**: You need a valid Google API key with the Gemini API enabled.
* **Tesseract OCR**: Must be installed on your system and preferably in your system's PATH.
    * Download and Install Tesseract: [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
* **Poppler**: Required by the `pdf2image` library to convert PDF files to images.
    * **Windows**: Download Poppler binaries. You'll need to add the `bin/` directory to your system's PATH. [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)
    * **macOS**: `brew install poppler`
    * **Linux (Debian/Ubuntu)**: `sudo apt-get install poppler-utils`
* Python libraries listed in `requirements.txt`.

## Setup Instructions

1.  **Clone the repository (if applicable) or download the script.**

2.  **Install Python Dependencies**:
    Open your terminal or command prompt and run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Tesseract OCR**:
    Follow the installation instructions for your operating system from the link provided in the "Requirements" section. Ensure that the Tesseract executable is in your system's PATH or update the path in the script.

4.  **Install Poppler**:
    Follow the installation instructions for your operating system from the links provided in the "Requirements" section. For Windows, ensure the Poppler `bin` directory is added to your PATH.

5.  **Configure Google API Key**:
    * Open the Python script (`your_script_name.py`).
    * Locate the line:
        ```python
        GOOGLE_API_KEY_MANUAL = "YOUR_API_KEY_HERE"
        ```
    * Replace `"YOUR_API_KEY_HERE"` with your actual Google API Key.

6.  **Configure Tesseract Path (if necessary)**:
    * If Tesseract is not automatically found (e.g., not in your system PATH), uncomment and update the following line in the script with the correct path to your Tesseract executable:
        ```python
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Windows example
        # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' # Linux example
        # pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract' # macOS (Apple Silicon) example
        ```
    * The script attempts to set a default Windows path. Adjust this if your installation directory is different or if you are on a different OS.

## Usage

1.  **Place your PDF file** (e.g., `input.pdf`) in the same directory as the Python script.
2.  If your PDF file has a different name, update the `pdf_to_process` variable in the script:
    ```python
    if __name__ == "__main__":
        # 3. SET THE PDF FILE TO PROCESS HERE
        pdf_to_process = "your_pdf_filename.pdf"
    ```
3.  **Run the script** from your terminal:
    ```bash
    python your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of your Python file).
4.  The script will print progress updates to the console.
5.  The final extracted and assembled answers will be saved in a `.txt` file named `[original_pdf_name]_answers.txt` in the same directory (e.g., `input_answers.txt`).

## Agent Overview

* **`DocumentIngestionAgent`**: Loads the PDF, converts each page to an image, and assesses whether the page is suitable for direct LLM processing or requires a hybrid OCR approach.
* **`LLMTextExtractionAgent`**: Directly extracts text from images using a Gemini vision model. Used for pages deemed suitable by the ingestion agent.
* **`HybridOCRAgent`**: Performs OCR using Tesseract on an image and then uses a Gemini LLM to correct and enhance the OCR output based on the image. Used for pages deemed less suitable for direct LLM.
* **`AnswerAssemblyAgent`**: Takes the text extracted from all pages, identifies questions, and consolidates their full answers. Uses a more advanced Gemini model for reasoning and structuring the final output.

## Configuration Notes

* **API Key**: The `GOOGLE_API_KEY_MANUAL` variable **must** be set for the script to function.
* **Tesseract Path**: If you encounter `TesseractNotFoundError`, ensure Tesseract is installed correctly and the path in `pytesseract.pytesseract.tesseract_cmd` is correctly set within the script, or that Tesseract is in your system's PATH.
* **Poppler Path**: If `pdf2image` fails, ensure Poppler is installed and its `bin` directory is in your system's PATH (especially on Windows).

## Troubleshooting

* **`Error: Please replace 'YOUR_API_KEY_HERE'...`**: You have not set your Google API Key in the script.
* **`Error: PDF file not found...`**: Ensure the PDF file name specified in `pdf_to_process` exists in the same directory as the script, or provide the correct path.
* **`TesseractNotFoundError`**:
    * Tesseract OCR is not installed.
    * Tesseract is not in your system PATH.
    * The `pytesseract.pytesseract.tesseract_cmd` path in the script is incorrect.
* **`Error converting PDF to images... Ensure Poppler is installed...`**:
    * Poppler is not installed.
    * The Poppler `bin` directory is not in your system PATH (especially relevant for Windows users).
* **LLM Errors (e.g., "No content generated, possibly due to safety filters")**:
    * The image content might have triggered Google's safety filters.
    * There might be an issue with your API key or billing for the Google Cloud project.
    * The image might be too blurry or contain no recognizable text for the LLM.

## License

This project is open-source. Please feel free to modify and use it. Consider adding a specific license (e.g., MIT, Apache 2.0) if you plan to distribute it widely.