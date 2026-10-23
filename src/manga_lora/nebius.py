import functools

import openai
from openai import AsyncOpenAI

from manga_lora.settings import settings


class MissingToken(RuntimeError):
    pass


@functools.cache
def client() -> AsyncOpenAI:
    if not settings.nebius_api_token:
        raise MissingToken("set NEBIUS_API_TOKEN in .env")
    return AsyncOpenAI(
        base_url=settings.base_url,
        api_key=settings.nebius_api_token,
        timeout=openai.Timeout(connect=10, read=settings.stall_timeout, write=30, pool=10),
        max_retries=0,
    )
