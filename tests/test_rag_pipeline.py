from src.rag_pipeline import generate_answer

def test_generate_answer_without_chunks():
    """The bot should not call Gemini when no policy chunks are retrieved."""

    result = generate_answer(
        question="What is the transfer limit?",
        chunks=[],
    )

    assert result["supported"] is False
    assert "do not provide enough information" in result["answer"]

def test_generate_answer_without_chunks_does_not_need_api_key(monkeypatch):
    """An empty retrieval result should return safely without an API key."""

    monkeypatch.setattr("src.rag_pipeline.GEMINI_API_KEY", None)

    result = generate_answer(
        question="What is the transfer limit?",
        chunks=[],
    )

    assert result["supported"] is False
    assert result["answer"]
    
import json

from src import rag_pipeline


def test_generate_answer_with_supported_response(monkeypatch):
    """A supported Gemini response should return its answer."""

    fake_response = type(
        "FakeResponse",
        (),
        {
            "text": json.dumps({
                "answer": "Block the card through the official bank app.",
                "supported": True,
            })
        },
    )()

    class FakeModels:
        def generate_content(self, **kwargs):
            return fake_response

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

        def close(self):
            pass

    monkeypatch.setattr(rag_pipeline.genai, "Client", FakeClient)
    monkeypatch.setattr(rag_pipeline, "GEMINI_API_KEY", "fake-test-key")

    chunks = [{
        "text": "Lost cards should be blocked through the official bank app.",
        "source": "sample_banking_policy.pdf",
        "page_number": 1,
        "score": 0.9,
    }]

    result = rag_pipeline.generate_answer(
        "What should I do if I lose my card?",
        chunks,
    )

    assert result["supported"] is True
    assert result["answer"] == "Block the card through the official bank app."


def test_generate_answer_with_unsupported_response(monkeypatch):
    """An unsupported Gemini response should be marked unsupported."""

    fake_response = type(
        "FakeResponse",
        (),
        {
            "text": json.dumps({
                "answer": "The available policy documents do not provide enough information to answer this question.",
                "supported": False,
            })
        },
    )()

    class FakeModels:
        def generate_content(self, **kwargs):
            return fake_response

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

        def close(self):
            pass

    monkeypatch.setattr(rag_pipeline.genai, "Client", FakeClient)
    monkeypatch.setattr(rag_pipeline, "GEMINI_API_KEY", "fake-test-key")

    chunks = [{
        "text": "The policy describes lost debit cards and minimum balances.",
        "source": "sample_banking_policy.pdf",
        "page_number": 1,
        "score": 0.8,
    }]

    result = rag_pipeline.generate_answer(
        "What is the maximum daily transfer limit?",
        chunks,
    )

    assert result["supported"] is False
    assert "do not provide enough information" in result["answer"]

def test_generate_answer_with_malformed_json(monkeypatch):
    """Invalid JSON should produce a safe fallback."""

    fake_response = type(
        "FakeResponse",
        (),
        {"text": "This is not valid JSON"},
    )()

    class FakeModels:
        def generate_content(self, **kwargs):
            return fake_response

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

        def close(self):
            pass

    monkeypatch.setattr(rag_pipeline.genai, "Client", FakeClient)
    monkeypatch.setattr(rag_pipeline, "GEMINI_API_KEY", "fake-test-key")

    chunks = [{
        "text": "Lost cards should be blocked immediately.",
        "source": "sample_banking_policy.pdf",
        "page_number": 1,
        "score": 0.9,
    }]

    result = rag_pipeline.generate_answer(
        "What should I do if I lose my card?",
        chunks,
    )

    assert result["supported"] is False
    assert "couldn't verify an answer" in result["answer"]

def test_answer_question_includes_sources_for_supported_answer(monkeypatch):
    """Supported answers should include their source and page."""

    chunks = [
        {
            "text": "Lost cards should be blocked immediately.",
            "source": "sample_banking_policy.pdf",
            "page_number": 1.0,
            "score": 0.9,
        }
    ]

    monkeypatch.setattr(
        rag_pipeline,
        "connect_to_pinecone",
        lambda: (None, "fake-index"),
    )
    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_chunks",
        lambda index, question, top_k=3: chunks,
    )
    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        lambda question, chunks: {
            "answer": "Block your card immediately.",
            "supported": True,
        },
    )

    result = rag_pipeline.answer_question("What if I lose my card?")

    assert result["answer"] == "Block your card immediately."
    assert result["sources"] == [
        {
            "source": "sample_banking_policy.pdf",
            "page_number": 1,
        }
    ]


def test_answer_question_omits_sources_for_unsupported_answer(monkeypatch):
    """Unsupported answers should not include sources."""

    chunks = [
        {
            "text": "The policy describes lost cards.",
            "source": "sample_banking_policy.pdf",
            "page_number": 1,
            "score": 0.8,
        }
    ]

    monkeypatch.setattr(
        rag_pipeline,
        "connect_to_pinecone",
        lambda: (None, "fake-index"),
    )
    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_chunks",
        lambda index, question, top_k=3: chunks,
    )
    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        lambda question, chunks: {
            "answer": (
                "The available policy documents do not provide "
                "enough information to answer this question."
            ),
            "supported": False,
        },
    )

    result = rag_pipeline.answer_question(
        "What is the maximum transfer limit?"
    )

    assert result["sources"] == []
    assert "do not provide enough information" in result["answer"]

def test_retrieve_chunks_returns_text_and_metadata(monkeypatch):
    """Retrieved chunks should include text, source, page, and score."""

    from types import SimpleNamespace

    fake_hit = SimpleNamespace(
        fields={
            "text → chunk_text": "Lost cards should be blocked immediately.",
            "source": "sample_banking_policy.pdf",
            "page_number": 1,
        },
        score=0.95,
    )

    fake_response = SimpleNamespace(
        result=SimpleNamespace(hits=[fake_hit])
    )

    class FakeIndex:
        def search(self, **kwargs):
            assert kwargs["namespace"] == rag_pipeline.NAMESPACE
            assert kwargs["top_k"] == 3
            assert kwargs["inputs"] == {
                "text": "What if I lose my card?"
            }
            return fake_response

    result = rag_pipeline.retrieve_chunks(
        FakeIndex(),
        "What if I lose my card?",
    )

    assert len(result) == 1
    assert result[0]["text"] == "Lost cards should be blocked immediately."
    assert result[0]["source"] == "sample_banking_policy.pdf"
    assert result[0]["page_number"] == 1
    assert result[0]["score"] == 0.95


def test_retrieve_chunks_returns_empty_list_when_no_hits():
    """No matching results should return an empty list."""

    from types import SimpleNamespace

    fake_response = SimpleNamespace(
        result=SimpleNamespace(hits=[])
    )

    class FakeIndex:
        def search(self, **kwargs):
            return fake_response

    result = rag_pipeline.retrieve_chunks(
        FakeIndex(),
        "What is the maximum transfer limit?",
    )

    assert result == []


def test_retrieve_chunks_skips_hits_without_text():
    """Records without usable policy text should be skipped."""

    from types import SimpleNamespace

    hits = [
        SimpleNamespace(
            fields={
                "text → chunk_text": "",
                "source": "empty.pdf",
                "page_number": 1,
            },
            score=0.9,
        ),
        SimpleNamespace(
            fields={
                "source": "missing_text.pdf",
                "page_number": 2,
            },
            score=0.8,
        ),
        SimpleNamespace(
            fields={
                "text → chunk_text": "The minimum balance is ₹5,000.",
                "source": "sample_banking_policy.pdf",
                "page_number": 1,
            },
            score=0.7,
        ),
    ]

    fake_response = SimpleNamespace(
        result=SimpleNamespace(hits=hits)
    )

    class FakeIndex:
        def search(self, **kwargs):
            return fake_response

    result = rag_pipeline.retrieve_chunks(
        FakeIndex(),
        "What is the minimum balance?",
    )

    assert len(result) == 1
    assert result[0]["text"] == "The minimum balance is ₹5,000."
