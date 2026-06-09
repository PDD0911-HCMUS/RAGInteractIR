import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Literal

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer


Role = Literal["system", "user", "assistant"]
RoleHistory = List[Tuple[Role, str]]
logger = logging.getLogger("rair.local_llm")


class LocalLLMService:
    """
    Local text-generation backend with the same generate_answer() surface used
    by OpenAIService. It is intended for instruction-tuned local models such as
    Gemma 3 so RAIR experiments can run without API credit.
    """

    _model = None
    _tokenizer = None
    _lock = threading.Lock()
    _loaded_model_name = None

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        local_files_only: Optional[bool] = None,
        max_new_tokens: int = 512,
        hf_token: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or self._resolve_device()
        self.dtype = dtype or os.environ.get("LOCAL_LLM_DTYPE", "auto")
        self.local_files_only = (
            os.environ.get("HF_LOCAL_FILES_ONLY", "1") != "0"
            if local_files_only is None
            else local_files_only
        )
        self.max_new_tokens = max_new_tokens
        self.hf_token = hf_token or self._load_huggingface_token()

    @staticmethod
    def _load_huggingface_token() -> Optional[str]:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token:
            return token.strip()

        key_path = Path(__file__).resolve().parents[1] / "huggingface.key"
        if key_path.is_file():
            value = key_path.read_text(encoding="utf-8").strip()
            return value or None

        return None

    @staticmethod
    def _resolve_device() -> str:
        configured = os.environ.get("LOCAL_LLM_DEVICE")
        if configured:
            return configured
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _torch_dtype(self):
        if self.dtype == "auto":
            return "auto"
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        if self.dtype == "float32":
            return torch.float32
        return "auto"

    def _load(self):
        with self._lock:
            if (
                self.__class__._model is not None
                and self.__class__._tokenizer is not None
                and self.__class__._loaded_model_name == self.model_name
            ):
                return self.__class__._tokenizer, self.__class__._model

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
                token=self.hf_token,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self._torch_dtype(),
                device_map="auto" if self.device == "cuda" else None,
                local_files_only=self.local_files_only,
                token=self.hf_token,
            )
            if self.device != "cuda":
                model = model.to(self.device)
            model.eval()

            self.__class__._tokenizer = tokenizer
            self.__class__._model = model
            self.__class__._loaded_model_name = self.model_name
            return tokenizer, model

    @staticmethod
    def _history_to_messages(
        user_prompt: str,
        history: Optional[RoleHistory],
        instructions: Optional[str],
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        for role, content in history or []:
            clean = str(content or "").strip()
            if clean:
                messages.append({"role": role, "content": clean})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _build_prompt(
        self,
        tokenizer: Any,
        user_prompt: str,
        history: Optional[RoleHistory],
        instructions: Optional[str],
    ) -> str:
        messages = self._history_to_messages(user_prompt, history, instructions)
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        rendered = []
        for message in messages:
            rendered.append(f"{message['role'].upper()}: {message['content']}")
        rendered.append("ASSISTANT:")
        return "\n\n".join(rendered)

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
        if model and model != self.model_name:
            self.model_name = model

        tokenizer, local_model = self._load()
        prompt = self._build_prompt(
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            history=history,
            instructions=instructions,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(local_model.device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        output_tokens = int(max_output_tokens or self.max_new_tokens)
        logger.info(
            "Local LLM generation start model=%s input_tokens=%d max_new_tokens=%d device=%s",
            self.model_name,
            input_tokens,
            output_tokens,
            local_model.device,
        )

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": output_tokens,
            "do_sample": (temperature or 0.0) > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature is not None and temperature > 0:
            generation_kwargs["temperature"] = temperature

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = local_model.generate(**inputs, **generation_kwargs)
        elapsed = time.perf_counter() - start

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        logger.info(
            "Local LLM generation done output_tokens=%d elapsed=%.2fs",
            int(generated_ids.shape[-1]),
            elapsed,
        )
        return text
