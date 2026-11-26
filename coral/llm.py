import json

import httpx

from .config import Settings


class ModelGatewayError(RuntimeError):
    pass


class ModelClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def enabled(self):
        return self.settings.coral_llm_mode == "deepseek"

    async def complete_json(self, system_message, user_message, fallback):
        if not self.enabled:
            return fallback, None
        key = self.settings.coral_llm_api_key.get_secret_value()
        if not key:
            raise ModelGatewayError("Online mode requires a DeepSeek API key.")
        url = str(self.settings.coral_llm_base_url).rstrip("/") + "/chat/completions"
        payload = {
            "model": self.settings.coral_llm_model,
            "response_format": {"type": "json_object"},
            "max_tokens": 8192,
            "thinking": {"type": self.settings.coral_llm_thinking},
            "messages": [
                {"role": "system", "content": system_message + " Return one JSON object."},
                {"role": "user", "content": user_message},
            ],
        }
        if self.settings.coral_llm_thinking == "disabled":
            payload["temperature"] = 0.2
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                response = await client.post(url, headers={"Authorization": f"Bearer {key}"}, json=payload)
                response.raise_for_status()
            completion = response.json()
            choice = completion["choices"][0]
            if choice.get("finish_reason") not in (None, "stop"):
                raise ValueError("The JSON response did not finish normally.")
            result = json.loads(choice["message"]["content"])
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
            raise ModelGatewayError(f"Model request or JSON parsing failed ({type(error).__name__}).") from error
        if not isinstance(result, dict):
            raise ModelGatewayError("The model must return a JSON object.")
        return result, {
            "provider": "DeepSeek",
            "requested_model": payload["model"],
            "served_model": completion.get("model"),
            "completion_id": completion.get("id"),
            "thinking": self.settings.coral_llm_thinking,
            "usage": {key: value for key, value in (completion.get("usage") or {}).items()
                      if key in {"prompt_tokens", "completion_tokens", "total_tokens"} and isinstance(value, int)},
        }
