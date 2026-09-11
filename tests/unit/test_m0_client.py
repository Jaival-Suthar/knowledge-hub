import httpx

from knowledge_hub.inference.client import InferenceClient


def test_m0_client_generate(monkeypatch) -> None:
    def fake_post(url, **kwargs):
        request = httpx.Request("POST", url, json=kwargs["json"])
        return httpx.Response(200, json={"text": "ok"}, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)
    assert InferenceClient("http://m0").generate("hello")["text"] == "ok"
