"""
Root-level pytest conftest.

This runs before any test module is imported/collected. We monkeypatch
transformers.pipeline here (at module import time, not inside a fixture)
so that when tests import backend.app, the QA/summarization pipelines
are lightweight stand-ins instead of triggering multi-GB model downloads
from the Hugging Face Hub. This keeps CI fast and network-independent.
"""
import transformers


def _fake_pipeline(task, *args, **kwargs):
    def _call(*call_args, **call_kwargs):
        if task == "question-answering":
            return {"answer": "mocked answer", "score": 1.0, "start": 0, "end": 1}
        if task == "summarization":
            return [{"summary_text": "mocked summary"}]
        return None
    return _call


transformers.pipeline = _fake_pipeline
