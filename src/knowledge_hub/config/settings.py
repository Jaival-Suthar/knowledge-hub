from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "knowledge-hub"
    app_env: str = "development"
    log_level: str = "INFO"
    m0_base_url: str = "http://localhost:8000"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_dimension: int = 384
    embedding_normalize: bool = True
    reranker_model_name: str = "BAAI/bge-reranker-base"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_hub"
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rerank_top_k: int = 5
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
