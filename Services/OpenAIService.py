import os
from pathlib import Path
import threading
import traceback
from typing import Optional, List, Tuple, Literal, Dict, Any

from openai import OpenAI

# You can change this default model later
DEFAULT_MODEL = "gpt-5.4-mini"

Role = Literal["system", "user", "assistant"]
RoleHistory = List[Tuple[Role, str]]


class OpenAIService:
    """
    OOP-style wrapper for OpenAI Responses API.

    Main features:
    - Lazy singleton-like shared client
    - Role-based history support
    - Text generation
    - Optional image input support for future VL usage
    - Clean response parsing
    """

    _client = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or self._load_api_key()

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Export it or create an OpenAI.key file."
            )

    @staticmethod
    def _load_api_key() -> Optional[str]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return api_key.strip()

        key_path = Path(__file__).resolve().parents[1] / "OpenAI.key"
        if key_path.is_file():
            return key_path.read_text(encoding="utf-8").strip()

        return None

    @classmethod
    def _get_or_create_shared_client(cls, api_key: str) -> OpenAI:
        """
        Lazy-initialize a shared OpenAI client.
        """
        if cls._client is None:
            with cls._lock:
                if cls._client is None:
                    try:
                        cls._client = OpenAI(api_key=api_key)
                    except Exception as e:
                        print(">>> OpenAI client init failed:", repr(e))
                        traceback.print_exc()
                        raise
        return cls._client

    def get_client(self) -> OpenAI:
        return self._get_or_create_shared_client(self.api_key)

    @staticmethod
    def normalize_history_for_openai(history: Optional[RoleHistory]) -> List[Dict[str, Any]]:
        """
        Convert role-based history into Responses API input format.

        Input:
            [
                ("system", "You are helpful"),
                ("user", "Hello"),
                ("assistant", "Hi"),
            ]

        Output:
            [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": "You are helpful"}]
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hi"}]
                }
            ]
        """
        if not history:
            return []

        items: List[Dict[str, Any]] = []

        for role, content in history:
            clean_text = (content or "").strip()
            if not clean_text:
                continue

            if role == "system":
                items.append(
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": clean_text}],
                    }
                )
            elif role == "user":
                items.append(
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": clean_text}],
                    }
                )
            elif role == "assistant":
                items.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": clean_text}],
                    }
                )
            else:
                raise ValueError(f"Unsupported role: {role}")

        return items

    @staticmethod
    def _response_to_text(response: Any) -> str:
        """
        Safely convert Responses API object to plain text.
        """
        try:
            if hasattr(response, "output_text") and response.output_text:
                return response.output_text.strip()
        except Exception:
            pass

        try:
            return str(response).strip()
        except Exception:
            return ""

    def _build_text_input(
        self,
        user_prompt: str,
        history: Optional[RoleHistory] = None,
    ) -> List[Dict[str, Any]]:
        items = self.normalize_history_for_openai(history)
        items.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            }
        )
        return items

    def _build_multimodal_input(
        self,
        user_prompt: str,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        history: Optional[RoleHistory] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build text + image input for vision-capable models.

        Use either:
        - image_url
        - image_base64 (data URL string is recommended)
        """
        if not image_url and not image_base64:
            raise ValueError("Either image_url or image_base64 must be provided.")

        items = self.normalize_history_for_openai(history)

        content: List[Dict[str, Any]] = [
            {"type": "input_text", "text": user_prompt}
        ]

        if image_url:
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_url,
                }
            )

        if image_base64:
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_base64,
                }
            )

        items.append(
            {
                "role": "user",
                "content": content,
            }
        )
        return items

    def generate_answer(
        self,
        user_prompt: str,
        history: Optional[RoleHistory] = None,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        store: bool = False,
    ) -> str:
        """
        Standard text-only generation.

        Parameters:
        - user_prompt: current user text
        - history: list of (role, content)
        - model: override default model
        - instructions: optional top-level instructions
        - temperature: optional sampling temperature
        - max_output_tokens: optional response length cap
        - store: whether to store response server-side
        """
        client = self.get_client()
        selected_model = model or self.model_name

        try:
            input_items = self._build_text_input(
                user_prompt=user_prompt,
                history=history,
            )

            kwargs: Dict[str, Any] = {
                "model": selected_model,
                "input": input_items,
                "store": store,
            }

            if instructions:
                kwargs["instructions"] = instructions

            if temperature is not None:
                kwargs["temperature"] = temperature

            if max_output_tokens is not None:
                kwargs["max_output_tokens"] = max_output_tokens

            response = client.responses.create(**kwargs)
            return self._response_to_text(response)

        except Exception as e:
            print(">>> generate_answer() failed:", repr(e))
            traceback.print_exc()
            raise

    def generate_answer_with_image_url(
        self,
        user_prompt: str,
        image_url: str,
        history: Optional[RoleHistory] = None,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        store: bool = False,
    ) -> str:
        """
        Vision input using image URL.
        """
        client = self.get_client()
        selected_model = model or self.model_name

        try:
            input_items = self._build_multimodal_input(
                user_prompt=user_prompt,
                image_url=image_url,
                history=history,
            )

            kwargs: Dict[str, Any] = {
                "model": selected_model,
                "input": input_items,
                "store": store,
            }

            if instructions:
                kwargs["instructions"] = instructions

            if temperature is not None:
                kwargs["temperature"] = temperature

            if max_output_tokens is not None:
                kwargs["max_output_tokens"] = max_output_tokens

            response = client.responses.create(**kwargs)
            return self._response_to_text(response)

        except Exception as e:
            print(">>> generate_answer_with_image_url() failed:", repr(e))
            traceback.print_exc()
            raise

    def generate_answer_with_image_base64(
        self,
        user_prompt: str,
        image_base64_data_url: str,
        history: Optional[RoleHistory] = None,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        store: bool = False,
    ) -> str:
        """
        Vision input using base64 data URL, for example:
        data:image/jpeg;base64,/9j/4AAQSk...

        Note:
        The caller should prepare the data URL string.
        """
        client = self.get_client()
        selected_model = model or self.model_name

        try:
            input_items = self._build_multimodal_input(
                user_prompt=user_prompt,
                image_base64=image_base64_data_url,
                history=history,
            )

            kwargs: Dict[str, Any] = {
                "model": selected_model,
                "input": input_items,
                "store": store,
            }

            if instructions:
                kwargs["instructions"] = instructions

            if temperature is not None:
                kwargs["temperature"] = temperature

            if max_output_tokens is not None:
                kwargs["max_output_tokens"] = max_output_tokens

            response = client.responses.create(**kwargs)
            return self._response_to_text(response)

        except Exception as e:
            print(">>> generate_answer_with_image_base64() failed:", repr(e))
            traceback.print_exc()
            raise

    def create_response(
        self,
        input_items: List[Dict[str, Any]],
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        store: bool = False,
    ) -> Any:
        """
        Low-level method if you want the full raw response object.
        Useful for advanced workflows.
        """
        client = self.get_client()
        selected_model = model or self.model_name

        try:
            kwargs: Dict[str, Any] = {
                "model": selected_model,
                "input": input_items,
                "store": store,
            }

            if instructions:
                kwargs["instructions"] = instructions

            if temperature is not None:
                kwargs["temperature"] = temperature

            if max_output_tokens is not None:
                kwargs["max_output_tokens"] = max_output_tokens

            return client.responses.create(**kwargs)

        except Exception as e:
            print(">>> create_response() failed:", repr(e))
            traceback.print_exc()
            raise
