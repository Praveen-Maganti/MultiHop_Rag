from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

def chunk_text(pages: List[Dict], document_id: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    formatted_chunks = []
    chunk_index = 0
    for page_data in pages:
        chunks = text_splitter.split_text(page_data["text"])
        for chunk in chunks:
            formatted_chunks.append({
                "chunk_id": f"{document_id}_{chunk_index}",
                "document_id": document_id,
                "text": chunk,
                "page": page_data["metadata"].get("page", 1),
                "section": "Unknown",
                "chunk_index": chunk_index
            })
            chunk_index += 1
            
    return formatted_chunks
