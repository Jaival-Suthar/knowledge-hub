from __future__ import annotations

from knowledge_hub.inference.client import InferenceClient


class M0Embedder:
    def __init__(self, client: InferenceClient) -> None:
        self.client = client

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(texts)
