import os
from pdf2image import convert_from_path
from PIL import Image
import google.generativeai as genai

# --- CONFIG ---
PDF_PATH = "input.pdf"
IMAGE_DIR = "images"
OUTPUT_DIR = "outputs"
POPPLER_PATH = r"C:\poppler\poppler-24.08.0\Library\bin"  # Update this to your poppler path
OLLAMA_PROMPT = "Extract the handwritten text exactly as it appears in the image, without correcting spelling or grammar."

# Hardcoded Gemini API key (⚠️ Only for testing)
GEMINI_API_KEY = "API"  # Replace with your actual API key

# --- SETUP ---
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")  # or "models/gemini-pro-vision"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- STEP 1: Convert PDF to images ---
print("Converting PDF to images...")
pages = convert_from_path(PDF_PATH, dpi=300, poppler_path=POPPLER_PATH)

image_paths = []
for idx, page in enumerate(pages):
    img_path = os.path.join(IMAGE_DIR, f"page_{idx+1}.jpg")
    page.convert("RGB").save(img_path, "JPEG", quality=95)
    image_paths.append(img_path)

print(f"Saved {len(image_paths)} pages to {IMAGE_DIR}/")

# --- STEP 2: Process images with Gemini ---
all_outputs = []
print("Processing images with Gemini...")

for img_path in image_paths:
    img_file = os.path.basename(img_path)
    print(f"> Processing {img_file}")
    try:
        pil_img = Image.open(img_path)
        response = model.generate_content(
            [OLLAMA_PROMPT, pil_img],
            generation_config={"temperature": 0.2}
        )
        page_output = response.text
        all_outputs.append(f"\n=== {img_file} ===\n{page_output.strip()}")

        # Save per-page output
        with open(os.path.join(OUTPUT_DIR, f"{img_file}.txt"), "w", encoding="utf-8") as f:
            f.write(page_output)

    except Exception as e:
        print(f"Error processing {img_file}: {str(e)}")

# --- STEP 3: Combine all outputs ---
combined_output_path = os.path.join(OUTPUT_DIR, "combined_output.txt")
with open(combined_output_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(all_outputs))

print(f"\n✅ Done. All outputs saved in '{OUTPUT_DIR}/'")


