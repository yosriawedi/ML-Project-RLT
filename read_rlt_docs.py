"""
Extract text from RLT documentation files
"""
import sys

# Try to extract from DOCX
try:
    from docx import Document
    doc = Document(r'c:\Users\DELL\Downloads\(No subject)\DESCRIPTION.docx')
    print("=" * 80)
    print("DESCRIPTION.docx CONTENT:")
    print("=" * 80)
    for para in doc.paragraphs:
        if para.text.strip():
            print(para.text)
    print("\n")
except Exception as e:
    print(f"Error reading DOCX: {e}")
    print("Installing python-docx...")

# Try to extract from PDF
try:
    import PyPDF2
    with open(r'c:\Users\DELL\Downloads\(No subject)\zhu2015.pdf', 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        print("=" * 80)
        print("zhu2015.pdf CONTENT:")
        print(f"Total pages: {len(pdf_reader.pages)}")
        print("=" * 80)
        for page_num in range(min(5, len(pdf_reader.pages))):  # First 5 pages
            page = pdf_reader.pages[page_num]
            print(f"\n--- Page {page_num + 1} ---")
            print(page.extract_text())
except Exception as e:
    print(f"Error reading PDF: {e}")
    print("Trying alternative PDF library...")
    try:
        import pdfplumber
        with pdfplumber.open(r'c:\Users\DELL\Downloads\(No subject)\zhu2015.pdf') as pdf:
            print("=" * 80)
            print("zhu2015.pdf CONTENT (via pdfplumber):")
            print(f"Total pages: {len(pdf.pages)}")
            print("=" * 80)
            for page_num in range(min(5, len(pdf.pages))):
                page = pdf.pages[page_num]
                print(f"\n--- Page {page_num + 1} ---")
                print(page.extract_text())
    except Exception as e2:
        print(f"Error with pdfplumber: {e2}")
