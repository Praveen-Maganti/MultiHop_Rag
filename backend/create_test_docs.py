"""
Creates 5 DOCX files that form a multi-hop retrieval chain:

Praveen (S101) → Project P201 → Company C301 → FinTech Solutions → Financial Technology
"""
from docx import Document
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "test_docs")
os.makedirs(OUT_DIR, exist_ok=True)

docs = {
    "student_praveen.docx": [
        "Name: Praveen",
        "Age: 21",
        "University: VIT-AP",
        "Year: 4th",
        "CGPA: 9.44",
        "Student ID: S101",
        "Project ID: P201",
    ],
    "project_p201.docx": [
        "Project ID: P201",
        "Project Name: Distributed URL Shortener",
        "Student ID: S101",
        "Company ID: C301",
        "Duration: 6 months",
        "Status: Completed",
    ],
    "company_c301.docx": [
        "Company ID: C301",
        "Company Name: FinTech Solutions",
        "Domain: Financial Technology",
        "Location: Hyderabad",
        "Founded: 2015",
        "Employees: 500",
    ],
    "domain_fintech.docx": [
        "Domain: Financial Technology",
        "Also Known As: FinTech",
        "Industry: Banking and Finance",
        "Regulation: RBI",
        "Examples: PhonePe, Paytm, Razorpay",
    ],
    "industry_banking.docx": [
        "Industry: Banking and Finance",
        "Regulator: Reserve Bank of India",
        "Sector: Financial Services",
        "Key Players: HDFC, ICICI, SBI",
        "Growth Rate: 12 percent per year",
    ],
}

for filename, lines in docs.items():
    doc = Document()
    doc.add_paragraph(" ".join(lines))
    path = os.path.join(OUT_DIR, filename)
    doc.save(path)
    print(f"Created: {path}")

print("\nAll 5 test documents created successfully.")
