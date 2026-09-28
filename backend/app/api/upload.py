from fastapi import APIRouter, UploadFile, File, HTTPException
import uuid
from app.utils.document_parser import extract_text

router = APIRouter()

documents_store = {}

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.docx', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported.")
    
    try:
        content = await file.read()
        extracted_pages = extract_text(content, file.filename)
        extracted_text = "\\n".join([p["text"] for p in extracted_pages])
        
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        from app.utils.chunking import chunk_text
        chunks = chunk_text(extracted_pages, doc_id)
        
        from app.retrieval.vector_store import initialize_vector_store
        initialize_vector_store(doc_id, chunks)
        
        documents_store[doc_id] = {
            "filename": file.filename,
            "text": extracted_text,
            "length": len(extracted_text)
        }
        
        return {
            "document_id": doc_id,
            "filename": file.filename,
            "message": "Document parsed, chunked, and indexed successfully",
            "length": len(extracted_text),
            "chunks_created": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Remove a document from memory (vector store + documents store)."""
    from app.retrieval.vector_store import vector_db
    
    removed = False
    if document_id in vector_db:
        del vector_db[document_id]
        removed = True
    if document_id in documents_store:
        del documents_store[document_id]
        removed = True
    
    if not removed:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")
    
    return {"message": f"Document {document_id} removed successfully."}


@router.get("/list")
async def list_documents():
    """List all currently indexed documents."""
    return {
        "documents": [
            {"document_id": doc_id, "filename": info["filename"]}
            for doc_id, info in documents_store.items()
        ]
    }

