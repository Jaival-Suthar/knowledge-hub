class InferenceClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str) -> dict:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
