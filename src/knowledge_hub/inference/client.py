from __future__ import annotations

import httpx


class InferenceClient:
    """M2 client for M0 inference; M2 never imports or runs the model itself."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> dict:
        response = httpx.post(
            f"{self.base_url}/v1/generate",
            json={"prompt": prompt},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/v1/embed",
            json={"texts": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise TypeError("M0 /v1/embed response missing embeddings")
        return embeddings
