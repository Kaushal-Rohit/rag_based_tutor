import PyPDF2
import sys

pdf_path = r"c:\Users\kaushal\Desktop\major 2\KaushalRohit_resume.pdf"
reader = PyPDF2.PdfReader(pdf_path)
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    sys.stdout.buffer.write(f"--- PAGE {i+1} ---\n".encode('utf-8'))
    sys.stdout.buffer.write(text.encode('utf-8', errors='replace'))
    sys.stdout.buffer.write(b"\n")
