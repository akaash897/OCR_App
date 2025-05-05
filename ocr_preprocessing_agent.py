# ocr_preprocessing_agent.py

import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from skimage.filters import threshold_local

class PreprocessingAgent:
    def __init__(self, dpi=300, output_dir='cleaned_pages'):
        self.dpi = dpi
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_pdf(self, pdf_path):
        pages = convert_from_path(pdf_path, dpi=self.dpi)
        results = []
        for idx, img in enumerate(pages):
            img_np = np.array(img)
            cleaned = self.adaptive_preprocess(img_np)
            out_path = os.path.join(self.output_dir, f'page_{idx+1:03}.png')
            cv2.imwrite(out_path, cleaned)
            results.append(out_path)
        return results

    def adaptive_preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        quality_report = self.analyze_image(gray)

        if quality_report['noisy']:
            gray = cv2.medianBlur(gray, 3)
        if quality_report['low_contrast']:
            gray = cv2.equalizeHist(gray)

        T = threshold_local(gray, 15, offset=10)
        binary = (gray > T).astype("uint8") * 255
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        if quality_report['skewed']:
            binary_bgr = self.deskew(binary_bgr)

        return self.crop_to_content(binary_bgr)

    def analyze_image(self, gray):
        stddev = np.std(gray)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        return {
            'low_contrast': stddev < 40,
            'noisy': edge_density > 0.12,
            'skewed': self.detect_skew(gray) > 1.0
        }

    def detect_skew(self, gray):
        coords = np.column_stack(np.where(gray < 255))
        if coords.size == 0:
            return 0.0
        angle = cv2.minAreaRect(coords)[-1]
        return abs(angle if angle <= 45 else 90 - angle)

    def deskew(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = np.column_stack(np.where(gray < 255))
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def crop_to_content(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return img[y:y+h, x:x+w]
        return img
