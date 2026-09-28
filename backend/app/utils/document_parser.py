import fitz
from docx import Document
import io

def parse_pdf(file_bytes: bytes) -> list:
    pages = []
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            pages.append({"text": page.get_text() + "\n", "metadata": {"page": page_num + 1}})
    except Exception as e:
        raise ValueError(f"Error parsing PDF: {e}")
    return pages

def parse_docx(file_bytes: bytes) -> list:
    pages = []
    try:
        doc = Document(io.BytesIO(file_bytes))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        pages.append({"text": text, "metadata": {"page": 1}})
    except Exception as e:
        raise ValueError(f"Error parsing DOCX: {e}")
    return pages

def parse_txt(file_bytes: bytes) -> list:
    try:
        return [{"text": file_bytes.decode('utf-8'), "metadata": {"page": 1}}]
    except Exception as e:
        raise ValueError(f"Error parsing TXT: {e}")

def extract_text(file_bytes: bytes, filename: str) -> list:
    ext = filename.split('.')[-1].lower()
    
    if ext == 'pdf':
        return parse_pdf(file_bytes)
    elif ext == 'docx':
        return parse_docx(file_bytes)
    elif ext == 'txt':
        return parse_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
