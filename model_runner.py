import os
from pdf2image import convert_from_path
from PIL import Image
import subprocess

# --- CONFIG ---
PDF_PATH = "input.pdf"
IMAGE_DIR = "images"
OUTPUT_DIR = "outputs"
MODEL_NAME = "llava:7b"  # or "minicpm-v"
OLLAMA_PROMPT = "Extract the handwritten text exactly as it appears in the image, without correcting spelling or grammar."

# Set Poppler path for Windows
POPPLER_PATH = r"C:\poppler\poppler-24.08.0\Library\bin"  # Adjust if needed

# Ensure directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- STEP 1: Convert PDF to images ---
print("Converting PDF to images...")
pages = convert_from_path(PDF_PATH, dpi=300, poppler_path=POPPLER_PATH)

def resize_image(img, max_size=1024):
    img = img.convert("RGB")  # ensure correct format
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    return img

for idx, page in enumerate(pages):
    resized = resize_image(page)
    img_path = os.path.join(IMAGE_DIR, f"page_{idx+1}.jpg")
    resized.save(img_path, "JPEG", quality=95)
print(f"Saved and resized {len(pages)} pages to {IMAGE_DIR}/")

# --- STEP 2: Run each image through the model ---
print("Processing images with VLM...")
for img_file in sorted(os.listdir(IMAGE_DIR)):
    img_path = os.path.join(IMAGE_DIR, img_file)
    cmd = [
        "ollama", "run", MODEL_NAME,
        OLLAMA_PROMPT, img_path
    ]
    print(f"> Processing {img_file}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output_path = os.path.join(OUTPUT_DIR, f"{img_file}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {img_file}:\n{e.stderr}")

print(f"\n✅ Done. Outputs saved in {OUTPUT_DIR}/")
