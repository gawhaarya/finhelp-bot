
from pathlib import Path
import re

from pypdf import PdfReader

# Find the root folder of our FinHelp Bot project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def extract_pages(pdf_path):
    """
    Extract text from a PDF, one page at a time.

    Returns a list of dictionaries containing:
    - source: PDF filename
    - page_number: page number, starting at 1
    - text: extracted page text
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The input file must be a PDF.")

    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""

        # Replace repeated whitespace and clean up the text.
        cleaned_text = re.sub(r"\s+", " ", raw_text).strip()

        if cleaned_text:
            pages.append({
                "source": pdf_path.name,
                "page_number": page_number,
                "text": cleaned_text,
            })

    if not pages:
        raise ValueError(
            "No extractable text was found. "
            "The PDF may be scanned or empty."
        )

    return pages

if __name__ == "__main__":
    pdf_file = DATA_DIR / "sample_banking_policy.pdf"

    extracted_pages = extract_pages(pdf_file)

    print(f"PDF: {pdf_file.name}")
    print(f"Pages containing text: {len(extracted_pages)}")

    for page in extracted_pages:
        print(f"\n--- Page {page['page_number']} ---")
        print(page["text"][:1000])