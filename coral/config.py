from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


DATA = Path(__file__).resolve().with_name("data")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    coral_data_dir: Path = Path("run")
    coral_corpus_dir: Path = Path("corpus")
    coral_llm_mode: Literal["offline", "deepseek"] = "offline"
    coral_cluster_priors: Path = DATA / "cluster_priors.json"
    coral_molecular_catalog: Path = DATA / "molecular_descriptors.csv"
    coral_bo_workers: int = Field(default=1, ge=1)
    coral_llm_base_url: HttpUrl = HttpUrl("https://api.deepseek.com")
    coral_llm_model: str = "deepseek-v4-pro"
    coral_llm_thinking: Literal["disabled", "enabled"] = "disabled"
    coral_llm_api_key: SecretStr = SecretStr("")

    @property
    def rag_db(self):
        return self.coral_data_dir / "rag.sqlite3"

    @property
    def workflows_dir(self):
        return self.coral_data_dir / "workflows"
