from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import re
from typing import Dict, List, Optional
import logging
import os
from datetime import datetime
import fitz  # PyMuPDF for PDF processing
import pdf2image
import tempfile
import numpy as np
from collections import Counter
import langdetect
from langdetect import detect_langs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Language PDF Word Counter API",
    description="API to count words in scanned PDF documents using OCR with support for English, Hindi, Arabic, and Spanish",
    version="2.0.0"
)

# CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Language configurations for Tesseract
LANGUAGE_CODES = {
    'english': 'eng',
    'hindi': 'hin',
    'arabic': 'ara',
    'spanish': 'spa'
}

# Combined language string for Tesseract
TESSERACT_LANG = '+'.join(LANGUAGE_CODES.values())  # 'eng+hin+ara+spa'

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Enhanced image preprocessing for better OCR accuracy
    """
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to grayscale for better OCR
        image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # Apply slight blur to reduce noise
        image = image.filter(ImageFilter.MedianFilter(size=3))
        
        # Convert to numpy array for advanced processing
        img_array = np.array(image)
        
        # Apply threshold to create binary image
        threshold = np.mean(img_array)
        img_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
        
        # Convert back to PIL Image
        image = Image.fromarray(img_array)
        
        # Resize if image is too small (OCR works better on larger images)
        width, height = image.size
        if width < 1000 or height < 1000:
            scale_factor = max(1000/width, 1000/height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        logger.warning(f"Error in image preprocessing: {e}")
        return image

def detect_languages_in_text(text: str) -> Dict:
    """
    Detect languages present in the text
    """
    try:
        if not text.strip():
            return {"detected_languages": [], "confidence": {}}
        
        # Use langdetect for language detection
        detected = detect_langs(text)
        
        language_mapping = {
            'en': 'english',
            'hi': 'hindi',
            'ar': 'arabic', 
            'es': 'spanish'
        }
        
        detected_languages = []
        confidence = {}
        
        for lang in detected:
            if lang.lang in language_mapping:
                lang_name = language_mapping[lang.lang]
                detected_languages.append(lang_name)
                confidence[lang_name] = round(lang.prob, 3)
        
        return {
            "detected_languages": detected_languages,
            "confidence": confidence
        }
    except Exception as e:
        logger.warning(f"Language detection error: {e}")
        return {"detected_languages": ["unknown"], "confidence": {}}

def count_words_multilang(text: str) -> Dict:
    """
    Enhanced word counting with multi-language support - accurate counting
    """
    # Preserve original text structure but normalize whitespace
    cleaned_text = re.sub(r'[ \t]+', ' ', text.strip())  # Only normalize spaces/tabs, keep newlines
    
    if not cleaned_text:
        return {
            "word_count": 0,
            "unique_word_count": 0,
            "character_count": 0,
            "character_count_no_spaces": 0,
            "text": "",
            "language_stats": {},
            "language_breakdown": {}
        }
    
    # Count ALL words using a comprehensive approach
    # English/Latin words (including Spanish accented characters)
    latin_words = re.findall(r'\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+\b', cleaned_text)
    
    # Hindi words (Devanagari script) - continuous Devanagari sequences
    hindi_words = re.findall(r'[\u0900-\u097F]+', cleaned_text)
    
    # Arabic words
    arabic_words = re.findall(r'[\u0600-\u06FF\u0750-\u077F]+', cleaned_text)
    
    # Total word count
    total_words = len(latin_words) + len(hindi_words) + len(arabic_words)
    
    # Unique word count (case-insensitive for Latin, preserve case for other scripts)
    all_words = []
    all_words.extend([word.lower() for word in latin_words])  # Case-insensitive for Latin
    all_words.extend(hindi_words)  # Case-sensitive for Hindi
    all_words.extend(arabic_words)  # Case-sensitive for Arabic
    unique_words = len(set(all_words))
    
    # Character counts
    character_count = len(cleaned_text)
    character_count_no_spaces = len(cleaned_text.replace(' ', ''))
    
    # Language breakdown analysis
    lines = cleaned_text.split('\n')
    current_section = None
    language_sections = {
        'english': [],
        'hindi': [],
        'arabic': [],
        'spanish': []
    }
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line is a language header
        line_lower = line.lower()
        if line_lower in ['english', 'hindi', 'arabic', 'spanish']:
            current_section = line_lower
            continue
        
        # Skip titles/headers that are clearly identifiers
        if current_section and len(line.split()) < 10:  # Likely a title
            language_sections[current_section].append(line)
            continue
        
        # Auto-detect language based on script for longer text
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', line))
        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F]', line))
        latin_chars = len(re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]', line))
        
        if hindi_chars > 0 and hindi_chars >= max(arabic_chars, latin_chars) * 0.5:
            language_sections['hindi'].append(line)
        elif arabic_chars > 0 and arabic_chars >= max(hindi_chars, latin_chars) * 0.5:
            language_sections['arabic'].append(line)
        elif current_section:
            language_sections[current_section].append(line)
        else:
            language_sections['english'].append(line)
    
    # Count words for each language section
    language_breakdown = {}
    for lang, text_lines in language_sections.items():
        if not text_lines:
            language_breakdown[lang] = {
                "word_count": 0,
                "character_count": 0,
                "character_count_no_spaces": 0,
                "text": ""
            }
            continue
            
        lang_text = ' '.join(text_lines)
        
        if lang == 'hindi':
            # Count Hindi words
            hindi_words_in_section = re.findall(r'[\u0900-\u097F]+', lang_text)
            # Count any Latin words mixed in
            latin_words_in_section = re.findall(r'\b[a-zA-Z]+\b', lang_text)
            section_word_count = len(hindi_words_in_section) + len(latin_words_in_section)
        elif lang == 'arabic':
            arabic_words_in_section = re.findall(r'[\u0600-\u06FF\u0750-\u077F]+', lang_text)
            section_word_count = len(arabic_words_in_section)
        else:
            # English/Spanish
            latin_words_in_section = re.findall(r'\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+\b', lang_text)
            section_word_count = len(latin_words_in_section)
        
        char_count = len(lang_text)
        char_count_no_spaces = len(lang_text.replace(' ', ''))
        
        language_breakdown[lang] = {
            "word_count": section_word_count,
            "character_count": char_count,
            "character_count_no_spaces": char_count_no_spaces,
            "text": lang_text[:200] + "..." if len(lang_text) > 200 else lang_text
        }
    
    # Character counts by script type
    english_chars = len(re.findall(r'[a-zA-Z]', cleaned_text))
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', cleaned_text))
    arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F]', cleaned_text))
    spanish_chars = len(re.findall(r'[áéíóúüñÁÉÍÓÚÜÑ]', cleaned_text))
    
    language_stats = {
        "english_characters": english_chars,
        "hindi_characters": hindi_chars,
        "arabic_characters": arabic_chars,
        "spanish_characters": spanish_chars,
        "english_words": language_breakdown['english']['word_count'],
        "hindi_words": language_breakdown['hindi']['word_count'],
        "arabic_words": language_breakdown['arabic']['word_count'],
        "spanish_words": language_breakdown['spanish']['word_count'],
        "total_latin_words": len(latin_words),
        "total_hindi_words": len(hindi_words),
        "total_arabic_words": len(arabic_words)
    }
    
    # Detect languages in the text
    lang_detection = detect_languages_in_text(cleaned_text)
    
    return {
        "word_count": total_words,
        "unique_word_count": unique_words,
        "character_count": character_count,
        "character_count_no_spaces": character_count_no_spaces,
        "text": cleaned_text,
        "language_stats": language_stats,
        "language_breakdown": language_breakdown,
        "detected_languages": lang_detection["detected_languages"],
        "language_confidence": lang_detection["confidence"]
    }

def extract_text_from_pdf_enhanced(pdf_bytes: bytes) -> Dict:
    """
    Enhanced PDF text extraction with multi-language OCR support
    """
    try:
        page_data = []
        total_text = ""
        
        # First, try to extract text directly (for text-based PDFs)
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = pdf_document.page_count
        
        # Check if PDF has extractable text
        sample_text = ""
        for page_num in range(min(3, total_pages)):
            page = pdf_document[page_num]
            page_text = page.get_text()
            sample_text += page_text
        
        pdf_document.close()
        
        # If we got meaningful text directly, use direct extraction
        if len(sample_text.strip()) > 50:
            logger.info("PDF contains extractable text, using direct extraction")
            
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(total_pages):
                page = pdf_document[page_num]
                page_text = page.get_text()
                
                # Enhanced word counting for this page
                page_stats = count_words_multilang(page_text)
                
                page_data.append({
                    "page_number": page_num + 1,
                    "word_count": page_stats["word_count"],
                    "character_count": page_stats["character_count"],
                    "character_count_no_spaces": page_stats["character_count_no_spaces"],
                    "language_stats": page_stats["language_stats"],
                    "detected_languages": page_stats["detected_languages"],
                    "language_confidence": page_stats["language_confidence"],
                    "extraction_method": "direct_text"
                })
                
                total_text += page_text + "\n\n"
            
            pdf_document.close()
            
        else:
            # Use enhanced OCR with multi-language support
            logger.info("PDF appears to be scanned, using enhanced multi-language OCR")
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                temp_pdf.write(pdf_bytes)
                temp_pdf_path = temp_pdf.name
            
            try:
                # Convert PDF to images with higher DPI for better OCR
                images = pdf2image.convert_from_path(
                    temp_pdf_path,
                    dpi=400,  # Increased DPI for better accuracy
                    fmt='PNG'
                )
                
                # Apply enhanced OCR to each page
                for i, image in enumerate(images):
                    logger.info(f"Processing page {i+1}/{len(images)} with enhanced multi-language OCR")
                    
                    # Preprocess image for better OCR accuracy
                    processed_image = preprocess_image(image)
                    
                    # Enhanced Tesseract configuration for multi-language OCR
                    custom_config = r'--oem 3 --psm 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{}|;:,.<>?/~`áéíóúüñÁÉÍÓÚÜÑ -c preserve_interword_spaces=1'
                    
                    # Apply multi-language OCR
                    page_text = pytesseract.image_to_string(
                        processed_image,
                        lang=TESSERACT_LANG,
                        config=custom_config
                    )
                    
                    # Enhanced word counting for this page
                    page_stats = count_words_multilang(page_text)
                    
                    page_data.append({
                        "page_number": i + 1,
                        "word_count": page_stats["word_count"],
                        "character_count": page_stats["character_count"],
                        "character_count_no_spaces": page_stats["character_count_no_spaces"],
                        "language_stats": page_stats["language_stats"],
                        "detected_languages": page_stats["detected_languages"],
                        "language_confidence": page_stats["language_confidence"],
                        "extraction_method": "enhanced_ocr"
                    })
                    
                    total_text += page_text + "\n\n"
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_pdf_path):
                    os.unlink(temp_pdf_path)
        
        return {
            "total_text": total_text,
            "page_data": page_data,
            "total_pages": len(page_data)
        }
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

def extract_text_from_image_enhanced(image_bytes: bytes) -> str:
    """
    Enhanced image text extraction with multi-language OCR support
    """
    try:
        # Open and preprocess image
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = preprocess_image(image)
        
        # Enhanced Tesseract configuration
        custom_config = r'--oem 3 --psm 3 -c preserve_interword_spaces=1'
        
        # Extract text using multi-language OCR
        extracted_text = pytesseract.image_to_string(
            processed_image,
            lang=TESSERACT_LANG,
            config=custom_config
        )
        
        return extracted_text
    
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def count_words_enhanced(text: str, page_data: List[Dict] = None) -> Dict:
    """
    Enhanced word counting with comprehensive multi-language statistics
    """
    # Get base statistics
    base_stats = count_words_multilang(text)
    
    if not text.strip():
        return base_stats
    
    # Additional calculations
    lines = [line for line in text.split('\n') if line.strip()]
    line_count = len(lines)
    
    # Calculate averages
    avg_words_per_line = round(base_stats["word_count"] / line_count, 2) if line_count > 0 else 0
    
    # Aggregate language statistics from all pages
    if page_data:
        total_pages = len(page_data)
        
        # Aggregate language stats across pages
        aggregated_lang_stats = {
            "english_characters": 0,
            "hindi_characters": 0,
            "arabic_characters": 0,
            "spanish_characters": 0,
            "english_words": 0,
            "hindi_words": 0,
            "arabic_words": 0,
            "spanish_words": 0
        }
        
        # Aggregate language breakdown
        aggregated_breakdown = {
            "english": {"word_count": 0, "character_count": 0, "character_count_no_spaces": 0},
            "hindi": {"word_count": 0, "character_count": 0, "character_count_no_spaces": 0},
            "arabic": {"word_count": 0, "character_count": 0, "character_count_no_spaces": 0},
            "spanish": {"word_count": 0, "character_count": 0, "character_count_no_spaces": 0}
        }
        
        all_detected_languages = set()
        language_confidence_aggregate = {}
        
        for page in page_data:
            if "language_stats" in page:
                for key, value in page["language_stats"].items():
                    if key in aggregated_lang_stats:
                        aggregated_lang_stats[key] += value
            
            if "language_breakdown" in page:
                for lang, stats in page["language_breakdown"].items():
                    if lang in aggregated_breakdown:
                        aggregated_breakdown[lang]["word_count"] += stats["word_count"]
                        aggregated_breakdown[lang]["character_count"] += stats["character_count"]
                        aggregated_breakdown[lang]["character_count_no_spaces"] += stats["character_count_no_spaces"]
            
            if "detected_languages" in page:
                all_detected_languages.update(page["detected_languages"])
            
            if "language_confidence" in page:
                for lang, conf in page["language_confidence"].items():
                    if lang in language_confidence_aggregate:
                        language_confidence_aggregate[lang] = max(language_confidence_aggregate[lang], conf)
                    else:
                        language_confidence_aggregate[lang] = conf
        
        # Calculate averages
        avg_words_per_page = round(base_stats["word_count"] / total_pages, 2) if total_pages > 0 else 0
        avg_chars_per_page = round(base_stats["character_count"] / total_pages, 2) if total_pages > 0 else 0
        
        per_page_stats = {
            "total_pages": total_pages,
            "average_words_per_page": avg_words_per_page,
            "average_characters_per_page": avg_chars_per_page,
            "page_breakdown": page_data,
            "aggregated_language_stats": aggregated_lang_stats,
            "aggregated_language_breakdown": aggregated_breakdown,
            "all_detected_languages": list(all_detected_languages),
            "language_confidence_summary": language_confidence_aggregate
        }
    else:
        per_page_stats = None
    
    # Build comprehensive result
    result = {
        "word_count": base_stats["word_count"],
        "unique_word_count": base_stats["unique_word_count"],
        "character_count": base_stats["character_count"],
        "character_count_no_spaces": base_stats["character_count_no_spaces"],
        "line_count": line_count,
        "average_words_per_line": avg_words_per_line,
        "language_statistics": base_stats["language_stats"],
        "language_breakdown": base_stats["language_breakdown"],
        "detected_languages": base_stats["detected_languages"],
        "language_confidence": base_stats["language_confidence"],
        "supported_languages": list(LANGUAGE_CODES.keys()),
        "extracted_text_preview": base_stats["text"][:300] + "..." if len(base_stats["text"]) > 300 else base_stats["text"]
    }
    
    if per_page_stats:
        result["per_page_statistics"] = per_page_stats
    
    return result
    # Additional calculations
    lines = [line for line in text.split('\n') if line.strip()]
    line_count = len(lines)
    
    # Calculate averages
    avg_words_per_line = round(base_stats["word_count"] / line_count, 2) if line_count > 0 else 0
    
    # Aggregate language statistics from all pages
    if page_data:
        total_pages = len(page_data)
        
        # Aggregate language stats across pages
        aggregated_lang_stats = {
            "english_characters": 0,
            "hindi_characters": 0,
            "arabic_characters": 0,
            "spanish_characters": 0,
            "latin_words": 0,
            "hindi_words": 0,
            "arabic_words": 0
        }
        
        all_detected_languages = set()
        language_confidence_aggregate = {}
        
        for page in page_data:
            if "language_stats" in page:
                for key, value in page["language_stats"].items():
                    if key in aggregated_lang_stats:
                        aggregated_lang_stats[key] += value
            
            if "detected_languages" in page:
                all_detected_languages.update(page["detected_languages"])
            
            if "language_confidence" in page:
                for lang, conf in page["language_confidence"].items():
                    if lang in language_confidence_aggregate:
                        language_confidence_aggregate[lang] = max(language_confidence_aggregate[lang], conf)
                    else:
                        language_confidence_aggregate[lang] = conf
        
        # Calculate averages
        avg_words_per_page = round(base_stats["word_count"] / total_pages, 2) if total_pages > 0 else 0
        avg_chars_per_page = round(base_stats["character_count"] / total_pages, 2) if total_pages > 0 else 0
        
        per_page_stats = {
            "total_pages": total_pages,
            "average_words_per_page": avg_words_per_page,
            "average_characters_per_page": avg_chars_per_page,
            "page_breakdown": page_data,
            "aggregated_language_stats": aggregated_lang_stats,
            "all_detected_languages": list(all_detected_languages),
            "language_confidence_summary": language_confidence_aggregate
        }
    else:
        per_page_stats = None
    
    # Build comprehensive result
    result = {
        "word_count": base_stats["word_count"],
        "character_count": base_stats["character_count"],
        "character_count_no_spaces": base_stats["character_count_no_spaces"],
        "line_count": line_count,
        "average_words_per_line": avg_words_per_line,
        "language_statistics": base_stats["language_stats"],
        "detected_languages": base_stats["detected_languages"],
        "language_confidence": base_stats["language_confidence"],
        "supported_languages": list(LANGUAGE_CODES.keys()),
        "extracted_text_preview": base_stats["text"][:300] + "..." if len(base_stats["text"]) > 300 else base_stats["text"]
    }
    
    if per_page_stats:
        result["per_page_statistics"] = per_page_stats
    
    return result

@app.get("/")
async def root():
    """
    Health check endpoint
    """
    return {
        "message": "Multi-Language PDF Word Counter API is running",
        "status": "healthy",
        "supported_formats": ["PDF", "PNG", "JPG", "JPEG", "TIFF", "BMP"],
        "supported_languages": list(LANGUAGE_CODES.keys()),
        "features": [
            "Multi-language OCR (English, Hindi, Arabic, Spanish)",
            "Enhanced image preprocessing",
            "Per-page language statistics",
            "Language detection and confidence scores",
            "Direct text extraction for searchable PDFs"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/count-words")
async def count_words_in_document(file: UploadFile = File(...)):
    """
    Enhanced endpoint to count words in uploaded PDF or image document with multi-language support
    """
    # Validate file type
    allowed_types = [
        "application/pdf", 
        "image/png", 
        "image/jpeg", 
        "image/jpg", 
        "image/tiff", 
        "image/bmp"
    ]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed types: PDF, PNG, JPG, JPEG, TIFF, BMP"
        )
    
    # Validate file size
    max_size = 25 * 1024 * 1024 if file.content_type == "application/pdf" else 10 * 1024 * 1024
    file_content = await file.read()
    
    if len(file_content) > max_size:
        max_size_mb = 25 if file.content_type == "application/pdf" else 10
        raise HTTPException(
            status_code=400,
            detail=f"File size too large. Maximum size is {max_size_mb}MB."
        )
    
    try:
        logger.info(f"Processing file: {file.filename} ({file.content_type}) with multi-language OCR")
        
        # Extract text based on file type
        if file.content_type == "application/pdf":
            pdf_result = extract_text_from_pdf_enhanced(file_content)
            extracted_text = pdf_result["total_text"]
            page_data = pdf_result["page_data"]
            file_type = "PDF"
            
            # Enhanced word counting with language statistics
            word_stats = count_words_enhanced(extracted_text, page_data)
        else:
            extracted_text = extract_text_from_image_enhanced(file_content)
            file_type = "Image"
            
            # Enhanced word counting without per-page statistics
            word_stats = count_words_enhanced(extracted_text)
        
        # Prepare comprehensive response
        response = {
            "success": True,
            "filename": file.filename,
            "file_type": file_type,
            "file_size_bytes": len(file_content),
            "file_size_mb": round(len(file_content) / (1024 * 1024), 2),
            "processing_timestamp": datetime.now().isoformat(),
            "ocr_languages_used": list(LANGUAGE_CODES.keys()),
            "tesseract_config": TESSERACT_LANG,
            "statistics": word_stats
        }
        
        logger.info(f"Successfully processed {file.filename}: {word_stats['word_count']} words found across {len(word_stats.get('detected_languages', []))} detected languages")
        return response
        
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@app.post("/count-words-simple")
async def count_words_simple(file: UploadFile = File(...)):
    """
    Simplified endpoint with basic multi-language statistics
    """
    allowed_types = [
        "application/pdf", 
        "image/png", 
        "image/jpeg", 
        "image/jpg", 
        "image/tiff", 
        "image/bmp"
    ]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Supported: PDF, PNG, JPG, JPEG, TIFF, BMP"
        )
    
    file_content = await file.read()
    
    try:
        # Extract text based on file type
        if file.content_type == "application/pdf":
            pdf_result = extract_text_from_pdf_enhanced(file_content)
            extracted_text = pdf_result["total_text"]
            page_data = pdf_result["page_data"]
            word_stats = count_words_enhanced(extracted_text, page_data)
        else:
            extracted_text = extract_text_from_image_enhanced(file_content)
            word_stats = count_words_enhanced(extracted_text)
        
        return {
            "filename": file.filename,
            "word_count": word_stats["word_count"],
            "character_count": word_stats["character_count"],
            "detected_languages": word_stats["detected_languages"],
            "language_confidence": word_stats["language_confidence"],
            "total_pages": word_stats.get("per_page_statistics", {}).get("total_pages", 1),
            "processing_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Enhanced health check with language support verification
    """
    try:
        # Test Tesseract installation
        tesseract_version = pytesseract.get_tesseract_version()
        
        # Test available languages
        available_langs = pytesseract.get_languages(config='')
        supported_langs = {}
        
        for lang_name, lang_code in LANGUAGE_CODES.items():
            supported_langs[lang_name] = lang_code in available_langs
        
        # Test PyMuPDF
        pymupdf_version = fitz.version[0]
        
        return {
            "status": "healthy",
            "tesseract_version": str(tesseract_version),
            "pymupdf_version": pymupdf_version,
            "available_languages": available_langs,
            "supported_languages": supported_langs,
            "tesseract_lang_config": TESSERACT_LANG,
            "supported_formats": ["PDF", "PNG", "JPG", "JPEG", "TIFF", "BMP"],
            "features": [
                "Multi-language OCR",
                "Enhanced image preprocessing", 
                "Language detection",
                "Per-page statistics",
                "Language-specific word counting"
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/languages")
async def get_language_info():
    """
    Endpoint to get detailed language support information
    """
    try:
        available_langs = pytesseract.get_languages(config='')
        
        language_info = {}
        for lang_name, lang_code in LANGUAGE_CODES.items():
            language_info[lang_name] = {
                "tesseract_code": lang_code,
                "available": lang_code in available_langs,
                "character_ranges": {
                    "english": "a-zA-Z",
                    "hindi": "\\u0900-\\u097F (Devanagari)",
                    "arabic": "\\u0600-\\u06FF, \\u0750-\\u077F",
                    "spanish": "a-zA-ZáéíóúüñÁÉÍÓÚÜÑ"
                }.get(lang_name, "Various")
            }
        
        return {
            "supported_languages": language_info,
            "tesseract_config": TESSERACT_LANG,
            "all_available_languages": available_langs,
            "installation_notes": {
                "hindi": "Requires tesseract-ocr-hin package",
                "arabic": "Requires tesseract-ocr-ara package", 
                "spanish": "Requires tesseract-ocr-spa package",
                "english": "Default with tesseract installation"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking language support: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """
    Enhanced startup event with language support verification
    """
    logger.info("=== Multi-Language PDF Word Counter API Starting ===")
    
    # Test Tesseract installation
    logger.info("Testing Tesseract installation...")
    try:
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract version: {version}")
        
        # Check available languages
        available_langs = pytesseract.get_languages(config='')
        logger.info(f"Available Tesseract languages: {available_langs}")
        
        # Check our required languages
        missing_langs = []
        for lang_name, lang_code in LANGUAGE_CODES.items():
            if lang_code in available_langs:
                logger.info(f"✓ {lang_name} ({lang_code}) support available")
            else:
                logger.warning(f"✗ {lang_name} ({lang_code}) support missing")
                missing_langs.append(f"{lang_name} ({lang_code})")
        
        if missing_langs:
            logger.warning(f"Missing language packages: {', '.join(missing_langs)}")
            logger.warning("Install missing packages with: apt-get install tesseract-ocr-[lang]")
        
    except Exception as e:
        logger.error(f"Tesseract test failed: {e}")
    
    # Test PyMuPDF
    logger.info("Testing PyMuPDF installation...")
    try:
        logger.info(f"PyMuPDF version: {fitz.version[0]}")
    except Exception as e:
        logger.error(f"PyMuPDF test failed: {e}")
    
    # Test additional dependencies
    try:
        import langdetect
        logger.info("✓ Language detection support available")
    except ImportError:
        logger.warning("✗ Language detection unavailable - install langdetect package")
    
    try:
        import numpy
        logger.info("✓ Advanced image processing available")
    except ImportError:
        logger.warning("✗ Advanced image processing unavailable - install numpy package")
    
    logger.info("=== Multi-Language API Ready ===")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)