import os
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from text_chunker import chunk_text

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file"""
    reader = PdfReader(file_path)
    text = ""
    
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    return text


def extract_text_from_url(url: str) -> str:
    """Extract visible text from a webpage"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    text = soup.get_text(separator="\n")
    return text


def load_all_pdfs(folder_path: str) -> dict:
    """
    Load all PDFs from a folder
    Returns dictionary: {filename: extracted_text}
    """
    documents = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Reading {filename}...")
            documents[filename] = extract_text_from_pdf(file_path)

    return documents


if __name__ == "__main__":
    docs = load_all_pdfs("data/funding_calls")

    all_chunks = []

    for filename, text in docs.items():
        chunks = chunk_text(text)
        for chunk in chunks:
            all_chunks.append({
                "source": filename,
                "content": chunk
            })

    print(f"Total chunks created: {len(all_chunks)}")