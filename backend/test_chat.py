import asyncio
from dotenv import load_dotenv
load_dotenv()

from fastapi import UploadFile
from app.api.upload import upload_document
from app.api.chat import chat, ChatRequest

async def main():
    with open("test_doc.txt", "rb") as f:
        file = UploadFile(filename="test_doc.txt", file=f)
        try:
            up_res = await upload_document(file)
            doc_id = up_res["document_id"]
            print("Uploaded:", doc_id)
            
            req = ChatRequest(document_id=doc_id, question="What was the main reason Project Alpha failed?")
            chat_res = await chat(req)
            print("Chat Success!")
        except Exception as e:
            print("ERROR:", str(e))
            import traceback
            traceback.print_exc()
            
if __name__ == "__main__":
    asyncio.run(main())
