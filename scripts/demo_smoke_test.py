"""
End-to-End Demo Smoke Test
==========================
Run this script before an interview or live demo to verify the upload and retrieval pipeline is working.

It does the following:
1. Generates a temporary PDF with a unique, known fact.
2. Uploads the PDF to the running server.
3. Polls the upload status until processing is complete.
4. Queries the API for that specific fact.
5. Verifies the answer contains the fact and cites the uploaded PDF.

Usage:
    python scripts/demo_smoke_test.py
"""

import os
import sys
import time
import uuid
import httpx
import fitz  # PyMuPDF

API_BASE_URL = "http://localhost:8000/api/v1"

def create_sample_pdf(filename: str, unique_fact: str):
    """Creates a minimal PDF with a unique fact using PyMuPDF."""
    print(f"[*] Generating sample PDF: {filename}...")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), f"Demo Fact Document.\n\n{unique_fact}\n\nThis is meant to verify the indexing pipeline.")
    doc.save(filename)
    doc.close()

def run_smoke_test():
    session_id = str(uuid.uuid4())
    unique_keyword = f"ZetaOmega-{uuid.uuid4().hex[:6]}"
    unique_fact = f"The secret launch code for the {unique_keyword} protocol is 99482."
    pdf_filename = "demo_smoke_test.pdf"

    print("========================================")
    print("[INFO] Adaptive RAG - Demo Smoke Test")
    print("========================================")
    print(f"Session ID: {session_id}")
    print(f"Unique Fact: {unique_fact}")
    print("----------------------------------------")

    try:
        # 1. Create PDF
        create_sample_pdf(pdf_filename, unique_fact)

        # 2. Upload PDF
        print("[*] Uploading PDF to API...")
        with open(pdf_filename, "rb") as f:
            files = {"file": (pdf_filename, f, "application/pdf")}
            data = {"session_id": session_id}
            response = httpx.post(f"{API_BASE_URL}/upload", files=files, data=data, timeout=60.0)
            
        if response.status_code != 202:
            print(f"[!] Upload failed with status {response.status_code}: {response.text}")
            sys.exit(1)
            
        data = response.json()
        job_id = data["job_id"]
        print(f"[+] Upload accepted. Job ID: {job_id}")

        # 3. Poll for completion
        print("[*] Polling for indexing completion (max 60s)...")
        start_time = time.time()
        job_ready = False
        while time.time() - start_time < 60:
            status_resp = httpx.get(f"{API_BASE_URL}/upload/{job_id}/status")
            status_data = status_resp.json()
            status = status_data["status"]
            
            if status == "ready":
                chunks = status_data.get("chunks_created", 0)
                print(f"[+] Indexing complete! Created {chunks} chunks.")
                job_ready = True
                break
            elif status == "failed":
                print(f"[!] Indexing failed: {status_data.get('error')}")
                sys.exit(1)
            else:
                print(f"    ...status: {status}")
                time.sleep(2)
                
        if not job_ready:
            print("[!] Timeout waiting for indexing to complete.")
            sys.exit(1)

        # 4. Query the unique fact
        print(f"[*] Querying the LLM for the unique keyword: '{unique_keyword}'...")
        query_payload = {
            "query": f"What is the secret launch code for the {unique_keyword} protocol?",
            "subject": "Physics", # Dummy subject to satisfy validation
            "session_id": session_id
        }
        query_resp = httpx.post(f"{API_BASE_URL}/query", json=query_payload, timeout=600.0)
        
        if query_resp.status_code != 200:
            print(f"[!] Query failed with status {query_resp.status_code}: {query_resp.text}")
            sys.exit(1)
            
        query_data = query_resp.json()
        answer = query_data["answer"]
        sources = query_data.get("sources", [])
        
        print("\n[+] Received Answer:")
        print(f"    {answer}")
        
        print("\n[+] Sources cited:")
        for s in sources:
            file_name = s.get('filename') or s.get('file_name', 'unknown_file')
            page_num = s.get('page_number', 'N/A')
            print(f"    - {file_name} (Page {page_num})")
            
        # 5. Verify results
        if "99482" not in answer:
            print("\n[!] VERIFICATION FAILED: The answer did not contain the expected fact.")
            sys.exit(1)
            
        if not any((s.get('filename') == pdf_filename or s.get('file_name') == pdf_filename) for s in sources):
            print("\n[!] VERIFICATION FAILED: The uploaded PDF was not cited in the sources.")
            sys.exit(1)

        print("\n========================================")
        print("[SUCCESS] SMOKE TEST PASSED!")
        print("   Upload, Indexing, and RAG Retrieval are fully operational.")
        print("   The system is safe to demo.")
        print("========================================")

    finally:
        # Cleanup
        if os.path.exists(pdf_filename):
            os.remove(pdf_filename)

if __name__ == "__main__":
    run_smoke_test()
