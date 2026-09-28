import asyncio
from fastapi import UploadFile
from app.api.upload import upload_document

async def main():
    with open("test_doc.txt", "rb") as f:
        file = UploadFile(filename="test_doc.txt", file=f)
        try:
            res = await upload_document(file)
            print("SUCCESS:", res)
        except Exception as e:
            print("ERROR:", str(e))
            
if __name__ == "__main__":
    asyncio.run(main())
