from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MANGA_LORA_", env_file=".env", extra="ignore")

    nebius_api_token: str = Field(default="", validation_alias="NEBIUS_API_TOKEN")
    base_url: str = "https://api.tokenfactory.nebius.com/v1"
    api_model: str = "moonshotai/Kimi-K3"
    stall_timeout: float = 60
    step_timeout: float = 600


settings = Settings()
