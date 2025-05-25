from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import io
import re
from typing import Dict, List
import logging
import os
from datetime import datetime
import fitz  # PyMuPDF for PDF processing
import pdf2image
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Word Counter API",
    description="API to count words in scanned PDF documents using OCR",
    version="1.0.0"
)

# CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def count_words_in_text(text: str) -> Dict:
    """
    Count words and characters in a given text
    """
    # Clean the text - remove extra whitespace and normalize
    cleaned_text = re.sub(r'\s+', ' ', text.strip())
    
    # Split into words (handling punctuation)
    words = re.findall(r'\b\w+\b', cleaned_text.lower())
    
    word_count = len(words)
    character_count = len(cleaned_text)
    character_count_no_spaces = len(cleaned_text.replace(' ', ''))
    
    return {
        "word_count": word_count,
        "character_count": character_count,
        "character_count_no_spaces": character_count_no_spaces,
        "text": cleaned_text
    }

def extract_text_from_pdf(pdf_bytes: bytes) -> Dict:
    """
    Extract text from PDF using OCR for scanned documents with per-page statistics
    """
    try:
        page_data = []
        total_text = ""
        
        # First, try to extract text directly (for text-based PDFs)
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = pdf_document.page_count
        
        # Check if PDF has extractable text
        sample_text = ""
        for page_num in range(min(3, total_pages)):  # Check first 3 pages
            page = pdf_document[page_num]
            page_text = page.get_text()
            sample_text += page_text
        
        pdf_document.close()
        
        # If we got meaningful text directly, use direct extraction
        if len(sample_text.strip()) > 50:  # Threshold for meaningful text
            logger.info("PDF contains extractable text, using direct extraction")
            
            # Re-open and extract text page by page
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(total_pages):
                page = pdf_document[page_num]
                page_text = page.get_text()
                
                # Count words and characters for this page
                page_stats = count_words_in_text(page_text)
                
                page_data.append({
                    "page_number": page_num + 1,
                    "word_count": page_stats["word_count"],
                    "character_count": page_stats["character_count"],
                    "character_count_no_spaces": page_stats["character_count_no_spaces"],
                    "extraction_method": "direct_text"
                })
                
                total_text += page_text + "\n\n"
            
            pdf_document.close()
            
        else:
            # If no meaningful text found, use OCR on images
            logger.info("PDF appears to be scanned, using OCR")
            
            # Convert PDF pages to images and apply OCR
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                temp_pdf.write(pdf_bytes)
                temp_pdf_path = temp_pdf.name
            
            try:
                # Convert PDF to images
                images = pdf2image.convert_from_path(
                    temp_pdf_path,
                    dpi=300,  # High resolution for better OCR
                    fmt='PNG'
                )
                
                # Apply OCR to each page
                for i, image in enumerate(images):
                    logger.info(f"Processing page {i+1}/{len(images)} with OCR")
                    
                    # Apply OCR
                    page_text = pytesseract.image_to_string(image)
                    
                    # Count words and characters for this page
                    page_stats = count_words_in_text(page_text)
                    
                    page_data.append({
                        "page_number": i + 1,
                        "word_count": page_stats["word_count"],
                        "character_count": page_stats["character_count"],
                        "character_count_no_spaces": page_stats["character_count_no_spaces"],
                        "extraction_method": "ocr"
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

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from image using OCR (Optical Character Recognition)
    """
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract text using Tesseract OCR
        extracted_text = pytesseract.image_to_string(image)
        
        return extracted_text
    
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def count_words(text: str, page_data: List[Dict] = None) -> Dict:
    """
    Count words in the extracted text and return detailed statistics
    """
    # Clean the text - remove extra whitespace and normalize
    cleaned_text = re.sub(r'\s+', ' ', text.strip())
    
    # Split into words (handling punctuation)
    words = re.findall(r'\b\w+\b', cleaned_text.lower())
    
    # Count statistics
    word_count = len(words)
    character_count = len(cleaned_text)
    character_count_no_spaces = len(cleaned_text.replace(' ', ''))
    line_count = len([line for line in text.split('\n') if line.strip()])
    
    # Get unique words
    unique_words = set(words)
    unique_word_count = len(unique_words)
    
    # Calculate average words per line
    avg_words_per_line = round(word_count / line_count, 2) if line_count > 0 else 0
    
    # Calculate per-page averages if page data is available
    per_page_stats = {}
    if page_data:
        total_pages = len(page_data)
        avg_words_per_page = round(word_count / total_pages, 2) if total_pages > 0 else 0
        avg_chars_per_page = round(character_count / total_pages, 2) if total_pages > 0 else 0
        
        per_page_stats = {
            "total_pages": total_pages,
            "average_words_per_page": avg_words_per_page,
            "average_characters_per_page": avg_chars_per_page,
            "page_breakdown": page_data
        }
    
    result = {
        "word_count": word_count,
        "unique_word_count": unique_word_count,
        "character_count": character_count,
        "character_count_no_spaces": character_count_no_spaces,
        "line_count": line_count,
        "average_words_per_line": avg_words_per_line,
        "extracted_text_preview": cleaned_text[:300] + "..." if len(cleaned_text) > 300 else cleaned_text
    }
    
    # Add per-page statistics if available
    if per_page_stats:
        result["per_page_statistics"] = per_page_stats
    
    return result

@app.get("/")
async def root():
    """
    Health check endpoint
    """
    return {
        "message": "PDF Word Counter API is running",
        "status": "healthy",
        "supported_formats": ["PDF", "PNG", "JPG", "JPEG", "TIFF", "BMP"],
        "features": ["Per-page statistics for PDFs", "OCR support", "Direct text extraction"],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/count-words")
async def count_words_in_document(file: UploadFile = File(...)):
    """
    Main endpoint to count words in uploaded PDF or image document
    
    Args:
        file: Uploaded file (PDF, PNG, JPG, JPEG, TIFF, BMP)
    
    Returns:
        Dictionary containing word count and text statistics with per-page breakdown for PDFs
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
    
    # Validate file size (max 25MB for PDFs, 10MB for images)
    max_size = 25 * 1024 * 1024 if file.content_type == "application/pdf" else 10 * 1024 * 1024
    file_content = await file.read()
    
    if len(file_content) > max_size:
        max_size_mb = 25 if file.content_type == "application/pdf" else 10
        raise HTTPException(
            status_code=400,
            detail=f"File size too large. Maximum size is {max_size_mb}MB."
        )
    
    try:
        logger.info(f"Processing file: {file.filename} ({file.content_type})")
        
        # Extract text based on file type
        if file.content_type == "application/pdf":
            pdf_result = extract_text_from_pdf(file_content)
            extracted_text = pdf_result["total_text"]
            page_data = pdf_result["page_data"]
            file_type = "PDF"
            
            # Count words with per-page statistics
            word_stats = count_words(extracted_text, page_data)
        else:
            extracted_text = extract_text_from_image(file_content)
            file_type = "Image"
            
            # Count words without per-page statistics
            word_stats = count_words(extracted_text)
        
        # Prepare response
        response = {
            "success": True,
            "filename": file.filename,
            "file_type": file_type,
            "file_size_bytes": len(file_content),
            "file_size_mb": round(len(file_content) / (1024 * 1024), 2),
            "processing_timestamp": datetime.now().isoformat(),
            "statistics": word_stats
        }
        
        logger.info(f"Successfully processed {file.filename}: {word_stats['word_count']} words found")
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
    Simplified endpoint that returns only essential information
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
            pdf_result = extract_text_from_pdf(file_content)
            extracted_text = pdf_result["total_text"]
            page_data = pdf_result["page_data"]
            word_stats = count_words(extracted_text, page_data)
        else:
            extracted_text = extract_text_from_image(file_content)
            word_stats = count_words(extracted_text)
        
        return {
            "filename": file.filename,
            "word_count": word_stats["word_count"],
            "character_count": word_stats["character_count"],
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
    Detailed health check endpoint
    """
    try:
        # Test Tesseract installation
        tesseract_version = pytesseract.get_tesseract_version()
        
        # Test PyMuPDF
        pymupdf_version = fitz.version[0]
        
        return {
            "status": "healthy",
            "tesseract_version": str(tesseract_version),
            "pymupdf_version": pymupdf_version,
            "supported_formats": ["PDF", "PNG", "JPG", "JPEG", "TIFF", "BMP"],
            "features": ["Per-page statistics", "OCR support", "Direct text extraction"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/docs")
async def get_docs():
    """
    API documentation endpoint
    """
    return {
        "title": "PDF Word Counter API Documentation",
        "endpoints": {
            "POST /count-words": "Upload PDF/image and get detailed word statistics with per-page breakdown",
            "POST /count-words-simple": "Upload PDF/image and get basic word count",
            "GET /health": "Check API health and dependencies",
            "GET /": "Basic API information"
        },
        "supported_formats": ["PDF", "PNG", "JPG", "JPEG", "TIFF", "BMP"],
        "max_file_sizes": {
            "PDF": "25MB",
            "Images": "10MB"
        },
        "new_features": {
            "per_page_statistics": "For PDFs: word count and character count per page",
            "page_breakdown": "Detailed statistics for each page including extraction method",
            "average_calculations": "Average words/characters per page"
        },
        "example_usage": {
            "curl": "curl -X POST 'https://your-api.railway.app/count-words' -H 'Content-Type: multipart/form-data' -F 'file=@document.pdf'"
        }
    }

@app.on_event("startup")
async def startup_event():
    """
    Startup event to log important information
    """
    logger.info("=== PDF Word Counter API Starting ===")
    logger.info("Testing Tesseract installation...")
    try:
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract version: {version}")
    except Exception as e:
        logger.error(f"Tesseract test failed: {e}")
    
    logger.info("Testing PyMuPDF installation...")
    try:
        logger.info(f"PyMuPDF version: {fitz.version[0]}")
    except Exception as e:
        logger.error(f"PyMuPDF test failed: {e}")
    
    logger.info("=== API Ready with Per-Page Statistics ===")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)