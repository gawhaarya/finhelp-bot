
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

#from pdf_processor import extract_pages
#from chunker import chunk_pages, validate_chunks
from src.pdf_processor import extract_pages
from src.chunker import chunk_pages, validate_chunks

# Find the project folders.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Load API keys from the project's .env file.
load_dotenv(PROJECT_ROOT / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "finhelp-bot"
NAMESPACE = "banking-policies"

# This must match the record field in your Pinecone schema.
TEXT_FIELD = "text → chunk_text"

def connect_to_pinecone():
    """Connect to the existing Pinecone index."""

    if (
        not PINECONE_API_KEY
        or PINECONE_API_KEY == "your_pinecone_key_here"
    ):
        raise ValueError(
            "PINECONE_API_KEY is missing. Check your .env file."
        )

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.index(name=INDEX_NAME)

    return pc, index

def load_and_chunk_pdf():
    """Extract the sample policy PDF and split it into chunks."""

    pdf_file = DATA_DIR / "sample_banking_policy.pdf"

    pages = extract_pages(pdf_file)

    chunks = chunk_pages(
        pages,
        chunk_size=500,
        overlap=100,
    )

    validate_chunks(chunks, chunk_size=500)

    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks prepared: {len(chunks)}")

    return chunks

def upload_chunks(index, chunks):
    """Upload text chunks and their source metadata to Pinecone."""

    records = []

    for chunk in chunks:
        record = {
            "_id": chunk["chunk_id"],
            TEXT_FIELD: chunk["text"],
            "source": chunk["source"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
        }

        records.append(record)

    response = index.upsert_records(
        namespace=NAMESPACE,
        records=records,
    )

    print("\nUpload request completed!")
    print(f"Records submitted: {response.record_count}")
    print(f"Namespace: {NAMESPACE}")


def search_chunks(index, question, top_k=3):
    """Find policy chunks most relevant to a question."""

    response = index.search(
        namespace=NAMESPACE,
        top_k=top_k,
        inputs={"text": question},
    )

    hits = response.result.hits

    print(f"\nQuestion: {question}")
    print(f"Retrieved chunks: {len(hits)}")

    for rank, hit in enumerate(hits, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Chunk ID: {hit.id}")
        print(f"Similarity score: {hit.score}")
        print(f"Source: {hit.fields.get('source')}")
        print(f"Page: {hit.fields.get('page_number')}")
        # print(f"Text: {hit.fields.get('chunk_text')}")
        retrieved_text = hit.fields.get("text → chunk_text")
        print(f"Text: {retrieved_text}")

if __name__ == "__main__":
    pc, index = connect_to_pinecone()

    print(f"Connected to index: {INDEX_NAME}")

    chunks = load_and_chunk_pdf()
    upload_chunks(index, chunks)
    
    search_chunks(
    index,
    "What should I do if I lose my debit card?",
    top_k=3,
)