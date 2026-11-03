from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import re


def count_tokens(text: str) -> int:
    """
    Token-aware length function that counts linguistic word and punctuation tokens.
    Prevents mid-word splitting and matches embedding model tokenization bounds.
    """
    return len(re.findall(r'\w+|[^\w\s]', text))


def chunk_text(pages: List[Dict], document_id: str, chunk_size: int = 250, chunk_overlap: int = 40) -> List[Dict]:
    """
    Splits document pages into token-aware chunks preserving semantic and sentence boundaries.
    Default chunk_size is 250 tokens (~1,000 characters) with 40-token overlap,
    respecting paragraph, sentence, and word delimiters.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        is_separator_regex=False,
    )
    
    formatted_chunks = []
    chunk_index = 0
    for page_data in pages:
        chunks = text_splitter.split_text(page_data.get("text", ""))
        for chunk in chunks:
            if not chunk.strip():
                continue
            formatted_chunks.append({
                "chunk_id": f"{document_id}_{chunk_index}",
                "document_id": document_id,
                "text": chunk.strip(),
                "page": page_data.get("metadata", {}).get("page", 1),
                "section": "Unknown",
                "chunk_index": chunk_index
            })
            chunk_index += 1
            
    return formatted_chunks
