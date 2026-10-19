from pathlib import Path

#from pdf_processor import extract_pages
from src.pdf_processor import extract_pages

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def chunk_pages(pages, chunk_size=500, overlap=100):
    """
    Split extracted PDF pages into overlapping text chunks.

    Each chunk retains its source filename and page number.
    Chunk positions are measured in characters.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        text = page["text"]
        source = page["source"]
        page_number = page["page_number"]

        chunk_index = 1

        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size]

            if not chunk_text.strip():
                continue

            chunks.append({
                "chunk_id": (
                    f"{source}-p{page_number}-c{chunk_index}"
                ),
                "source": source,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text,
            })

            chunk_index += 1

            # No need to create another chunk after reaching the end.
            if start + chunk_size >= len(text):
                break

    return chunks


def validate_chunks(chunks, chunk_size=500):
    """Check that generated chunks have valid text and metadata."""

    if not chunks:
        raise ValueError("Validation failed: no chunks were generated.")

    required_fields = {
        "chunk_id",
        "source",
        "page_number",
        "chunk_index",
        "text",
    }

    seen_ids = set()

    for chunk in chunks:
        missing = required_fields - chunk.keys()

        if missing:
            raise ValueError(f"Missing fields: {missing}")

        if not chunk["text"].strip():
            raise ValueError("A chunk contains empty text.")

        if len(chunk["text"]) > chunk_size:
            raise ValueError("A chunk exceeds the configured size.")

        if chunk["chunk_id"] in seen_ids:
            raise ValueError("Duplicate chunk ID detected.")

        seen_ids.add(chunk["chunk_id"])

    print("Chunk validation passed!")
    print(f"Validated chunks: {len(chunks)}")
    print(f"Unique chunk IDs: {len(seen_ids)}")

if __name__ == "__main__":
    pdf_file = DATA_DIR / "sample_banking_policy.pdf"

    pages = extract_pages(pdf_file)
    chunks = chunk_pages(pages, chunk_size=500, overlap=100)
    validate_chunks(chunks, chunk_size=500)

    print(f"Extracted pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks:
        print(
            f"\n--- Chunk {chunk['chunk_index']} "
            f"(Page {chunk['page_number']}) ---"
        )
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Characters: {len(chunk['text'])}")
        print(chunk["text"])