from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    sample_text = "This is a sample document for testing chunking. " * 100
    chunks = chunk_text(sample_text)
    print(f"Created {len(chunks)} chunks.")