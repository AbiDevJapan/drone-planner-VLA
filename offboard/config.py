"""Local .env-based settings for the OFFBOARD Ollama client only.

Nothing about connection details belongs in mission-state messages or visual logs.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
from urllib.parse import urlsplit

from dotenv import load_dotenv


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str
    model: str
    timeout_s: float = 45.0
    temperature: float = 0.1
    num_ctx: int = 8192
    num_predict: int = 180
    think: bool = False

    @classmethod
    def from_env(cls, env_file: str | None = None) -> 'OllamaSettings':
        # Actual process environment overrides values from the .env file.
        load_dotenv(dotenv_path=env_file if env_file is not None else Path.cwd() / ".env", override=False)
        url = os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434').rstrip('/')
        parts = urlsplit(url)
        if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password or parts.path not in ('', '/') or parts.query or parts.fragment:
            raise ValueError('OLLAMA_BASE_URL must be a bare http(s) origin, e.g. http://192.168.1.50:11434')
        # Explicit port is optional: default Ollama port convention applies only if omitted.
        if not (model := os.getenv('OLLAMA_MODEL', 'qwen3:8b').strip()):
            raise ValueError('OLLAMA_MODEL cannot be empty')
        timeout = float(os.getenv('OLLAMA_TIMEOUT_S', '45'))
        temperature = float(os.getenv('OLLAMA_TEMPERATURE', '0.1'))
        ctx = int(os.getenv('OLLAMA_NUM_CTX', '8192'))
        pred = int(os.getenv('OLLAMA_NUM_PREDICT', '180'))
        think_str = os.getenv('OLLAMA_THINK', 'false').strip().lower()
        if think_str not in ('true', 'false', '1', '0', 'yes', 'no'):
            raise ValueError('OLLAMA_THINK must be true or false')
        if timeout <= 0 or not 0 <= temperature <= 2 or ctx < 1024 or pred < 32:
            raise ValueError('Invalid timeout/temperature/context/generation settings')
        return cls(url, model, timeout, temperature, ctx, pred, think_str in ('true','1','yes'))
