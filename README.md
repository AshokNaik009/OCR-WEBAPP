# 📄 PDF Word Counter API & Web Application

A powerful, cloud-hosted solution for extracting text and counting words from PDF documents and images using advanced OCR (Optical Character Recognition) technology. Features per-page analysis, multiple file format support, and a beautiful web interface.

## 🚀 Live Demo

- **🌐 Web Application**: [https://word-counter-webapp-production.up.railway.app/](https://word-counter-webapp-production.up.railway.app/)
- **🔧 API Documentation**: [https://ocr-trial-project-production.up.railway.app/docs](https://ocr-trial-project-production.up.railway.app/docs)
- **📊 API Base URL**: `https://ocr-trial-project-production.up.railway.app`

## ✨ Features

### 🔍 Advanced Text Extraction
- **Smart PDF Processing**: Automatically detects text-based vs scanned PDFs
- **OCR Technology**: Uses Tesseract OCR for scanned documents
- **Multiple Formats**: Supports PDF, PNG, JPG, JPEG, TIFF, BMP
- **High Accuracy**: 300 DPI processing for optimal OCR results

### 📊 Comprehensive Statistics
- **Word Count**: Total and unique word counts
- **Character Analysis**: With and without spaces
- **Per-Page Breakdown**: Individual statistics for each PDF page
- **Line Count**: Total number of text lines
- **Average Calculations**: Words/characters per page

### 🎨 Beautiful Web Interface
- **Drag & Drop Upload**: Intuitive file upload experience
- **Real-time Processing**: Live progress indicators
- **Animated Results**: Smooth number animations
- **Responsive Design**: Works on all devices
- **Error Handling**: User-friendly error messages

### 🛡️ Enterprise Ready
- **File Validation**: Size and type restrictions
- **CORS Support**: Cross-origin request enabled
- **Health Monitoring**: Built-in health check endpoints
- **Scalable**: Cloud-hosted on Railway
- **Fast Processing**: Optimized for performance

## 🔧 API Endpoints

### Main Endpoints

| Method | Endpoint | Description | Max File Size |
|--------|----------|-------------|---------------|
| `GET` | `/` | API health check and information | - |
| `POST` | `/count-words` | Full analysis with per-page breakdown | 25MB (PDF), 10MB (Images) |
| `POST` | `/count-words-simple` | Basic word count only | 25MB (PDF), 10MB (Images) |
| `GET` | `/health` | Detailed health check | - |
| `GET` | `/docs` | Interactive API documentation | - |

### Supported File Types
- **PDF**: `application/pdf`
- **Images**: `image/png`, `image/jpeg`, `image/jpg`, `image/tiff`, `image/bmp`

## 📋 API Usage Examples

### Using cURL

```bash
# Basic word count
curl -X POST "https://ocr-trial-project-production.up.railway.app/count-words" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@document.pdf"
```

### Using Python

```python
import requests

# Upload and analyze document
def analyze_document(file_path):
    url = "https://ocr-trial-project-production.up.railway.app/count-words"
    
    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        data = response.json()
        stats = data['statistics']
        
        print(f"📄 File: {data['filename']}")
        print(f"📊 Total Words: {stats['word_count']:,}")
        print(f"📝 Characters: {stats['character_count']:,}")
        
        # Per-page statistics (for PDFs)
        if 'per_page_statistics' in stats:
            pages = stats['per_page_statistics']
            print(f"📑 Total Pages: {pages['total_pages']}")
            print(f"📈 Avg Words/Page: {pages['average_words_per_page']}")
            
            for page in pages['page_breakdown']:
                method = page['extraction_method']
                print(f"   Page {page['page_number']}: {page['word_count']} words ({method})")
    
    return response.json()

# Usage
result = analyze_document("my_document.pdf")
```

### Using JavaScript

```javascript
// Frontend JavaScript example
async function analyzeDocument(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(
            'https://ocr-trial-project-production.up.railway.app/count-words',
            {
                method: 'POST',
                body: formData
            }
        );
        
        const data = await response.json();
        
        console.log('Analysis Results:', data);
        console.log('Word Count:', data.statistics.word_count);
        
        // Display per-page statistics
        if (data.statistics.per_page_statistics) {
            const pages = data.statistics.per_page_statistics;
            console.log(`Document has ${pages.total_pages} pages`);
            console.log(`Average: ${pages.average_words_per_page} words/page`);
        }
        
        return data;
    } catch (error) {
        console.error('Error:', error);
    }
}
```

### Using Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const fetch = require('node-fetch');

async function analyzeDocument(filePath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    
    const response = await fetch(
        'https://ocr-trial-project-production.up.railway.app/count-words',
        {
            method: 'POST',
            body: form
        }
    );
    
    return await response.json();
}
```

## 📊 Response Format

### Full Analysis Response (`/count-words`)

```json
{
  "success": true,
  "filename": "document.pdf",
  "file_type": "PDF",
  "file_size_bytes": 2048576,
  "file_size_mb": 1.95,
  "processing_timestamp": "2025-05-25T11:30:45.123456",
  "statistics": {
    "word_count": 1250,
    "unique_word_count": 485,
    "character_count": 7892,
    "character_count_no_spaces": 6420,
    "line_count": 156,
    "average_words_per_line": 8.01,
    "extracted_text_preview": "This is a preview of the extracted text...",
    "per_page_statistics": {
      "total_pages": 5,
      "average_words_per_page": 250.0,
      "average_characters_per_page": 1578.4,
      "page_breakdown": [
        {
          "page_number": 1,
          "word_count": 245,
          "character_count": 1523,
          "character_count_no_spaces": 1245,
          "extraction_method": "direct_text"
        },
        {
          "page_number": 2,
          "word_count": 298,
          "character_count": 1687,
          "character_count_no_spaces": 1398,
          "extraction_method": "ocr"
        }
      ]
    }
  }
}
```

### Simple Response (`/count-words-simple`)

```json
{
  "filename": "document.pdf",
  "word_count": 1250,
  "character_count": 7892,
  "total_pages": 5,
  "processing_timestamp": "2025-05-25T11:30:45.123456"
}
```

## 🛠️ Technical Stack

### Backend (API)
- **Framework**: FastAPI (Python)
- **OCR Engine**: Tesseract OCR
- **PDF Processing**: PyMuPDF (fitz), pdf2image
- **Image Processing**: Pillow (PIL)
- **Deployment**: Railway Cloud Platform
- **CORS**: Enabled for cross-origin requests

### Frontend (Web App)
- **Technologies**: HTML5, CSS3, Vanilla JavaScript
- **UI Framework**: Custom responsive design
- **Animations**: CSS transitions and JavaScript
- **File Upload**: Drag & drop with validation
- **Deployment**: Railway Static Hosting

## 🚀 Deployment Architecture

```
┌─────────────────┐    ┌──────────────────┐
│   Frontend      │    │     Backend      │
│   Railway       │───▶│     Railway      │
│   Static Site   │    │   FastAPI + OCR  │
└─────────────────┘    └──────────────────┘
```

## 📈 Performance & Limits

| Metric | Value |
|--------|-------|
| **Max PDF Size** | 25 MB |
| **Max Image Size** | 10 MB |
| **Processing Time** | 2-30 seconds (depends on size) |
| **OCR Resolution** | 300 DPI |
| **Supported Languages** | English (extensible) |
| **Concurrent Requests** | Auto-scaling |

## 🔒 Security & Privacy

- **File Validation**: Strict file type and size checking
- **Temporary Processing**: Files are not permanently stored
- **Memory Management**: Automatic cleanup after processing
- **HTTPS**: All communications encrypted
- **No Data Retention**: Documents are processed and discarded

## 🐛 Error Handling

The API provides comprehensive error responses:

```json
{
  "detail": "File size too large. Maximum size is 25MB.",
  "status_code": 400
}
```

Common error scenarios:
- **400**: Invalid file type or size
- **422**: Missing or malformed request data
- **500**: Processing errors (OCR failures, corrupted files)

## 📊 Health Monitoring

Check API health and dependencies:

```bash
curl https://ocr-trial-project-production.up.railway.app/health
```

Response includes:
- Tesseract OCR version
- PyMuPDF version
- System status
- Supported formats

## 🤝 Integration Examples

### WordPress Plugin
```php
function analyze_pdf($file_path) {
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, 'https://ocr-trial-project-production.up.railway.app/count-words');
    curl_setopt($ch, CURLOPT_POST, 1);
    curl_setopt($ch, CURLOPT_POSTFIELDS, [
        'file' => new CURLFile($file_path)
    ]);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    
    $response = curl_exec($ch);
    curl_close($ch);
    
    return json_decode($response, true);
}
```

### React Component
```jsx
import { useState } from 'react';

function DocumentAnalyzer() {
    const [results, setResults] = useState(null);
    
    const analyzeFile = async (file) => {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(
            'https://ocr-trial-project-production.up.railway.app/count-words',
            { method: 'POST', body: formData }
        );
        
        const data = await response.json();
        setResults(data);
    };
    
    return (
        <div>
            <input 
                type="file" 
                onChange={(e) => analyzeFile(e.target.files[0])}
                accept=".pdf,.png,.jpg,.jpeg"
            />
            {results && (
                <div>
                    <h3>Analysis Results</h3>
                    <p>Words: {results.statistics.word_count}</p>
                    <p>Pages: {results.statistics.per_page_statistics?.total_pages}</p>
                </div>
            )}
        </div>
    );
}
```

## 📞 Support & Contact

- **Issues**: Please report bugs via GitHub issues
- **Features**: Feature requests welcome
- **Documentation**: Full API docs available at `/docs` endpoint
- **Status**: Check system status at `/health` endpoint

## 📜 License

This project is available for educational and commercial use. Please ensure compliance with OCR and text processing regulations in your jurisdiction.

---

**Built with ❤️ using FastAPI, Tesseract OCR, and Railway**

*Last updated: May 25, 2025*