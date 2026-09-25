from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    typesafe_api_key: str | None = None
    typesafe_url: str = "https://api.typesafe.ai/v1/systemone"
    typesafe_model: str = "jev-latest"
    voicevox_url: str = "http://127.0.0.1:50021"
    sudachi_dict_path: Path | None = None
    jmdict_path: Path | None = None
