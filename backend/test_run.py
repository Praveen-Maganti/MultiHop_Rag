import requests
import time
import os
import subprocess

BASE_URL = "http://localhost:8000"

def run_test():
    print("1. Creating test document...")
    doc_content = """
    Project Alpha was launched in March 2023. It had a budget of $5 million.
    The main reason for Project Alpha's failure was severe operational delays caused by a supply chain disruption.
    The primary database timeout was set to 30 seconds.
    However, a later audit found that the database timeout was actually set to 60 seconds, which caused cascading failures.
    The company migrated from monolith to microservices to improve scalability.
    """
    
    with open("test_doc.txt", "w") as f:
        f.write(doc_content)
        
    print("2. Uploading test document...")
    with open("test_doc.txt", "rb") as f:
        response = requests.post(f"{BASE_URL}/api/documents/upload", files={"file": f})
        
    if response.status_code != 200:
        print("Upload failed:", response.text)
        return
        
    doc_id = response.json()["document_id"]
    print(f"Document uploaded successfully! ID: {doc_id}")
    
    print("3. Running experiment (this will take a while due to rate limiting sleeps)...")
    questions = [
        "What was the main reason Project Alpha failed?",
        "What was the database timeout?",
        "Why did the company change architecture?"
    ]
    
    exp_response = requests.post(
        f"{BASE_URL}/api/experiments/run", 
        json={"document_id": doc_id, "questions": questions}
    )
    
    if exp_response.status_code != 200:
        print("Experiment failed:", exp_response.text)
        return
        
    print(f"Experiment completed! ID: {exp_response.json()['experiment_id']}")
    
    print("4. Generating report...")
    # Run the generate_report.py script in the same venv
    result = subprocess.run(["venv\\Scripts\\python", "generate_report.py"], capture_output=True, text=True)
    print(result.stdout)
    
    if os.path.exists("research_report.md"):
        print("5. Report contents:")
        with open("research_report.md", "r") as f:
            print(f.read())
            
if __name__ == "__main__":
    run_test()
