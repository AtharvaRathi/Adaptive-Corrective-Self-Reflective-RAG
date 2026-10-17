"""
src/config.py
─────────────
Centralized configuration management using pydantic-settings.
All values are read from environment variables or the .env file.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration resolved from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider ──────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", description="Groq API key")
    tavily_api_key: str = Field(default="", description="Tavily search API key")

    # ── Model Selection ───────────────────────────────────────────────────────
    llm_model: str = Field(
        default="qwen/qwen3.8-27b",
        description="Groq-hosted chat model (e.g. qwen/qwen3.8-27b)",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model (runs locally)",
    )

    # ── Document Chunking ─────────────────────────────────────────────────────
    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=50, ge=0)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retriever_dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    retriever_sparse_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    top_k_retrieval: int = Field(default=5, ge=1, le=20)

    # ── CRAG Loop Control ─────────────────────────────────────────────────────
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum CRAG retry iterations before forcing web search",
    )

    # ── Quality Thresholds ─────────────────────────────────────────────────────
    hallucination_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    faithfulness_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("retriever_dense_weight", "retriever_sparse_weight", mode="before")
    @classmethod
    def _coerce_float(cls, v: object) -> float:
        return float(v)

    @property
    def retriever_weights(self) -> list[float]:
        """Return [dense_weight, sparse_weight] normalised to sum to 1."""
        total = self.retriever_dense_weight + self.retriever_sparse_weight
        if total == 0:
            return [0.5, 0.5]
        return [
            self.retriever_dense_weight / total,
            self.retriever_sparse_weight / total,
        ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
