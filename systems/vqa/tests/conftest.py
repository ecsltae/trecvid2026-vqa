"""VQA test fixtures."""
import pytest
from trec_shared.ollama_client import OllamaClient


def _ollama_available(model="llava:13b"):
    try:
        client = OllamaClient(model=model)
        return client.is_available()
    except Exception:
        return False


@pytest.fixture
def client():
    """OllamaClient fixture — skips test if Ollama/llava not available."""
    if not _ollama_available("llava:13b"):
        pytest.skip("Ollama llava:13b not available")
    return OllamaClient(model="llava:13b", temperature=0.1)


@pytest.fixture
def qwen_client():
    """qwen3:32b client for text-only VQA tests."""
    if not _ollama_available("qwen3:32b"):
        pytest.skip("Ollama qwen3:32b not available")
    return OllamaClient(model="qwen3:32b", temperature=0.1)
