"""
Central configuration — loaded once at startup.

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict



class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 3306
    user: str = ""
    password: str = ""
    database: str = ""
    table: str = ""


class EmbeddingConfig(BaseModel):
    candidate_id: str = "id"
    fields: list[str] = ["title", "author", "synopsis", "subjects"]
    template: str = "{title}. By {authors}. {synopsis}. Topics: {subjects}"


class ModelConfig(BaseModel):
    name: str
    path: str


class IndexConfig(BaseModel):
    storage_dir: str = "indexes/"
    batch_size: int = 256



class AppConfig(BaseModel):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    models: list[ModelConfig] = Field(default_factory=list)
    index: IndexConfig = Field(default_factory=IndexConfig)

    def model_by_name(self, name: str) -> ModelConfig:
        for m in self.models:
            if m.name == name:
                return m
        available = [m.name for m in self.models]
        raise KeyError(f"Model '{name}' not found. Available: {available}")

    def build_text(self, row: dict[str, Any]) -> str:
        """
        Apply the embedding template to a DB row dict.
        Fields that are None/empty are replaced with an empty string.
        """
        safe = {f: (row.get(f) or "") for f in self.embedding.fields}
        return self.embedding.template.format_map(safe).strip()






_config: AppConfig | None = None
config_path: str = "config/config_dev.yaml"

def load_config(path: str | None = None) -> AppConfig:
    global _config

    load_dotenv()

    yaml_path = Path(path or os.environ.get("EMBLORE_CONFIG_PATH", config_path))

    raw: dict[str, Any] = {}
    if yaml_path.exists():
        with yaml_path.open() as f:
            raw = yaml.safe_load(f) or {}

    cfg = AppConfig.model_validate(raw)

    env_pw = os.environ.get("EMBLORE_DB__PASSWORD")
    env_user = os.environ.get("EMBLORE_DB__USER")
    env_database = os.environ.get("EMBLORE_DB__DATABASE")
    env_table = os.environ.get("EMBLORE_DB__TABLE")

    if env_pw:
        cfg.database.password = env_pw
    if env_user:
        cfg.database.user = env_user
    if env_database:
        cfg.database.database = env_database
    if env_table:
        cfg.database.table = env_table

    _config = cfg
    return cfg


def get_config() -> AppConfig:
    if _config is None:
        return load_config()
    return _config
