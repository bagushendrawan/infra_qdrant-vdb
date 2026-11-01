from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QDRANT_VDB_", env_file=".env", extra="ignore")
    data_path: str = "./data/qdrant"
    environment: str = "local"
    timeout: float = Field(default=120, gt=0)
    startup_timeout: float = Field(default=60, gt=0)
    max_bytes: int = Field(default=67108864, gt=0)
