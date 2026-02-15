from pydantic_ai.models import Model, ModelRequest, ModelResponse
from typing import AsyncIterator, Any, Optional
import httpx
import json

class OllamaModel(Model):
    """
    Custom Model implementation for Ollama using its OpenAI-compatible API.
    Implements all required abstract methods/properties from pydantic_ai.models.Model.
    """

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434/v1"):
        self._model_name = model_name          # Private backing attribute (writable)
        self._base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=60.0)

    @property
    def model_name(self) -> str:
        """Required abstract property: returns the model identifier."""
        return self._model_name

    @property
    def base_url(self) -> str:
        """Required abstract property: returns the base URL."""
        return self._base_url

    @property
    def system(self) -> str:
        """Required abstract property: returns the provider/system name."""
        return "ollama"

    async def request(
        self,
        request: ModelRequest,
        model_settings: Optional[Any] = None,
        request_parameters: Optional[Any] = None,
    ) -> ModelResponse:
        """
        Handle chat request to Ollama.
        """
        messages = []
        for msg in request.messages:
            role = msg.role
            content = msg.content
            messages.append({"role": role, "content": content})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }

        if request.tools:
            payload["tools"] = [tool.to_dict() for tool in request.tools]

        try:
            resp = await self.client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            message = choice["message"]
            content = message["content"]

            parsed = None
            if request.output_type:
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    pass

            return ModelResponse(
                content=content,
                parsed=parsed,
                usage=data.get("usage", {}),
                finish_reason=choice.get("finish_reason"),
            )
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")

    async def stream_request(self, request: ModelRequest, **kwargs) -> AsyncIterator[ModelResponse]:
        resp = await self.request(request, **kwargs)
        yield resp

    def customize_request_parameters(self, request_parameters: Any) -> Any:
        return request_parameters
    

if __name__ == "__main__":
    model = OllamaModel(model_name="codellama:34b")
    print("Custom Ollama model created successfully")
    print("model_name:", model.model_name)
    print("base_url:", model.base_url)
    print("system:", model.system)