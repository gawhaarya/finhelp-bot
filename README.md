# FinHelp Bot : Banking Policy Q&A Assistant

FinHelp Bot is a Retrieval-Augmented Generation (RAG) application that answers questions about banking policies using information retrieved from policy documents.

The project combines semantic search with a Large Language Model (LLM) to generate context-grounded answers and provide source references. When the available policy context does not contain enough information, the application returns a fallback response instead of intentionally inventing a banking rule.

## Features

- **PDF processing:** Extracts text from banking policy PDFs.
- **Text chunking:** Splits documents into smaller, overlapping chunks for retrieval.
- **Semantic search:** Uses Pinecone to retrieve relevant policy content.
- **LLM-powered answers:** Uses Google Gemini to generate answers based on retrieved context.
- **Source references:** Displays source documents and page numbers for supported answers.
- **Fallback handling:** Handles missing context and unsupported questions.
- **Interactive interface:** Provides a chat interface built with Streamlit.
- **Automated tests:** Uses pytest to test retrieval, answer handling, and source metadata.

## Architecture

```text
Banking Policy PDF
        |
        v
  PDF Text Extraction
        |
        v
  Text Chunking
        |
        v
  Pinecone Vector Database
        |
        v
    User Question
        |
        v
  Semantic Retrieval
        |
        v
 Retrieved Policy Context
        |
        v
   Google Gemini LLM
        |
        v
 Answer + Source References
        |
        v
   Streamlit Chat UI
```

## Technology Stack

- Python
- Streamlit
- Pinecone
- Google Gemini API (`google-genai`)
- PyPDF
- python-dotenv
- pytest

## Project Structure

```text
finhelp-bot/
├── data/
│   └── sample_banking_policy.pdf
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py
│   ├── chunker.py
│   ├── vector_store.py
│   └── rag_pipeline.py
├── tests/
│   └── test_rag_pipeline.py
├── app.py
├── requirements.txt
├── .env
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.12 or a compatible Python version
- A Pinecone account and API key
- A Google Gemini API key

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd finhelp-bot
```

Replace the placeholder with your actual repository URL after publishing the project.

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root with the following variable names:

```text
PINECONE_API_KEY=your_pinecone_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholder values with your own credentials. Never commit your `.env` file or publish API keys.

### 5. Configure Pinecone

Create a Pinecone index named `finhelp-bot` in the region and configuration expected by the application. Configure the index for the embedding model and integrated-embedding schema used by the project.

The current application expects the text field `text → chunk_text` and uses the `banking-policies` namespace.

### 6. Prepare the policy data

Place the sample policy PDF at:

```text
data/sample_banking_policy.pdf
```

Run the vector-store ingestion script from the project root to extract, chunk, and upload the policy content:

```powershell
python -m src.vector_store
```

### 7. Run the application

```powershell
streamlit run app.py
```
Open the local URL printed in the terminal.

## Example Questions

The included synthetic banking policy covers topics such as:

- What should I do if I lose my debit card?
- What is the minimum average monthly balance for a savings account?
- Is there a fee for falling below the required minimum balance?
- What should I do if an ATM transaction fails but my account is debited?
- What should I do if someone asks me for my PIN or OTP?

Questions about information absent from the policy, such as an unspecified transfer limit, should receive a fallback response.

## Running Tests

Run the automated test suite from the project root:

```powershell
python -m pytest -v
```

The current test suite covers retrieval behavior, supported and unsupported answer handling, malformed model responses, and source metadata. External services are mocked in these unit tests, so they do not prove that live model responses are always accurate.

## Limitations and Future Improvements

- The included banking policy is synthetic demonstration data, not an official bank policy.
- Answer quality depends on document coverage, retrieval relevance, and LLM behavior.
- Current unit tests do not replace end-to-end testing against live services.
- Further improvements could include retrieval relevance thresholds, stronger grounding evaluation, logging, API error handling, and additional security testing.

## Security

- Keep API keys in environment variables.
- Do not commit `.env` or other secret-bearing files.
- Do not use this demonstration application as a source of authoritative financial advice.

## Disclaimer

FinHelp Bot is an educational portfolio project. The sample policy is fictional and must not be treated as a real bank's terms, fees, or procedures.