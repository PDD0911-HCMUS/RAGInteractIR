import base64
import json
import mimetypes
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Services.OpenAIService import OpenAIService


class TargetObservationService:
    """
    Generate image-grounded observations for the simulated user.

    The RAIR system still receives only retrieved candidate evidence. These
    observations are attached to the hidden simulated user context so the user
    can behave more like someone looking at the actual target image.
    """

    _local_processor = None
    _local_model = None
    _local_model_key = None
    _local_lock = threading.Lock()

    def __init__(
        self,
        provider: str = "none",
        model_name: str = "gpt-4o-mini",
        image_root: Optional[str] = None,
        max_output_tokens: int = 512,
        device: str = "cuda",
        dtype: str = "bfloat16",
        local_files_only: bool = True,
    ) -> None:
        self.provider = (provider or "none").lower()
        self.model_name = model_name
        self.image_root = Path(image_root).expanduser() if image_root else None
        self.max_output_tokens = max_output_tokens
        self.device = device
        self.dtype = dtype
        self.local_files_only = local_files_only
        self._openai: Optional[OpenAIService] = None
        self._cache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _safe_json_loads(text: str) -> Dict[str, Any]:
        cleaned = str(text or "").strip().removeprefix("\ufeff").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            value = json.loads(cleaned)
        except Exception:
            decoder = json.JSONDecoder()
            start = cleaned.find("{")
            if start < 0:
                raise
            value, _ = decoder.raw_decode(cleaned[start:])

        if not isinstance(value, dict):
            raise ValueError(f"Target observation must be a JSON object: {value!r}")
        return value

    @classmethod
    def _observation_from_text(cls, text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```(?:json)?|```", "", str(text or ""), flags=re.IGNORECASE).strip()
        lines = [
            re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip(" ;,.")
            for line in cleaned.splitlines()
        ]
        facts = [line for line in lines if line]
        if not facts and cleaned:
            facts = [part.strip(" ;,.") for part in re.split(r"[.;]\s+", cleaned) if part.strip(" ;,.")]
        caption = facts[0] if facts else cleaned[:180]
        return {
            "caption": caption,
            "visual_facts": facts[:12],
            "uncertain_facts": [],
        }

    @staticmethod
    def _clean_list(values: Any, limit: int = 12) -> List[str]:
        if not isinstance(values, list):
            return []
        cleaned: List[str] = []
        seen = set()
        for value in values:
            text = re.sub(r"\s+", " ", str(value or "")).strip(" ;,.")
            key = text.lower()
            if text and key not in seen:
                cleaned.append(text)
                seen.add(key)
            if len(cleaned) >= limit:
                break
        return cleaned

    def _candidate_roots(self) -> List[Path]:
        roots: List[Path] = []
        if self.image_root:
            roots.append(self.image_root)

        for env_name in ("RAIR_IMAGE_ROOT", "RAIR_DATASETS_PATH"):
            value = os.environ.get(env_name)
            if value:
                roots.append(Path(value).expanduser())

        project_root = Path(__file__).resolve().parents[1]
        roots.extend([project_root / "datasets", project_root])

        deduped: List[Path] = []
        seen = set()
        for root in roots:
            key = str(root)
            if key not in seen:
                deduped.append(root)
                seen.add(key)
        return deduped

    def resolve_image_path(self, image_path: Any) -> Optional[Path]:
        raw = str(image_path or "").strip()
        if not raw:
            return None

        path = Path(raw)
        if path.is_absolute() and path.is_file():
            return path

        variants = [
            Path(raw),
            Path("MSCOCO") / raw,
            Path("mscoco") / raw,
            Path("COCO") / raw,
            Path("coco") / raw,
        ]
        for root in self._candidate_roots():
            for variant in variants:
                candidate = root / variant
                if candidate.is_file():
                    return candidate
        return None

    @staticmethod
    def _image_to_data_url(path: Path) -> str:
        mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"

    def _openai_service(self) -> OpenAIService:
        if self._openai is None:
            self._openai = OpenAIService(model_name=self.model_name)
        return self._openai

    def _torch_dtype(self):
        import torch

        value = (self.dtype or "auto").lower()
        if value in {"auto", "none"}:
            return "auto"
        if value in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if value in {"fp16", "float16", "half"}:
            return torch.float16
        if value in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported local VLM dtype: {self.dtype}")

    def _device_map(self):
        value = (self.device or "auto").lower()
        if value == "auto":
            return "auto"
        if value in {"cuda", "gpu"}:
            return {"": 0}
        if value.startswith("cuda:"):
            return {"": int(value.split(":", 1)[1])}
        if value == "cpu":
            return {"": "cpu"}
        return "auto"

    def _load_local_model(self) -> Tuple[Any, Any]:
        key = (
            self.model_name,
            str(self.device),
            str(self.dtype),
            bool(self.local_files_only),
        )
        cls = self.__class__
        if cls._local_model is None or cls._local_processor is None or cls._local_model_key != key:
            with cls._local_lock:
                if cls._local_model is None or cls._local_processor is None or cls._local_model_key != key:
                    from transformers import AutoProcessor

                    try:
                        from transformers import AutoModelForImageTextToText

                        model_cls = AutoModelForImageTextToText
                    except ImportError:
                        from transformers import AutoModelForVision2Seq

                        model_cls = AutoModelForVision2Seq

                    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                    processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        token=token,
                        local_files_only=self.local_files_only,
                    )
                    model = model_cls.from_pretrained(
                        self.model_name,
                        token=token,
                        local_files_only=self.local_files_only,
                        torch_dtype=self._torch_dtype(),
                        device_map=self._device_map(),
                    )
                    model.eval()
                    cls._local_processor = processor
                    cls._local_model = model
                    cls._local_model_key = key

        return cls._local_processor, cls._local_model

    def _generate_local_observation(self, image_path: Path, prompt: str) -> str:
        from PIL import Image
        import torch

        processor, model = self._load_local_model()
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        if hasattr(processor, "apply_chat_template"):
            text = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            text = prompt

        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        first_param = next(model.parameters())
        inputs = {
            key: value.to(first_param.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0

        generate_kwargs = {
            **inputs,
            "max_new_tokens": self.max_output_tokens,
            "do_sample": False,
        }
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
            generate_kwargs["pad_token_id"] = tokenizer.eos_token_id

        with torch.no_grad():
            output = model.generate(**generate_kwargs)

        if input_len:
            output = output[:, input_len:]
        if hasattr(processor, "batch_decode"):
            return processor.batch_decode(output, skip_special_tokens=True)[0].strip()
        return str(output)

    def generate_with_image(self, sample: Dict[str, Any], prompt: str) -> str:
        if self.provider in {"", "none", "off", "disabled"}:
            raise ValueError("Target image generation requires a vision provider.")
        if self.provider not in {"openai", "local", "local_vlm"}:
            raise ValueError(f"Unsupported target image generation provider: {self.provider}")

        image_path = self.resolve_image_path(sample.get("image_path"))
        if image_path is None:
            raise FileNotFoundError(f"target image not found: {sample.get('image_path')}")

        if self.provider == "openai":
            return self._openai_service().generate_answer_with_image_base64(
                user_prompt=prompt,
                image_base64_data_url=self._image_to_data_url(image_path),
                model=self.model_name,
                temperature=0.0,
                max_output_tokens=self.max_output_tokens,
                store=False,
            )
        return self._generate_local_observation(image_path=image_path, prompt=prompt)

    def observe(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.provider in {"", "none", "off", "disabled"}:
            return {}
        if self.provider not in {"openai", "local", "local_vlm"}:
            raise ValueError(f"Unsupported target observation provider: {self.provider}")

        image_path = self.resolve_image_path(sample.get("image_path"))
        if image_path is None:
            return {
                "provider": self.provider,
                "model": self.model_name,
                "error": f"target image not found: {sample.get('image_path')}",
            }

        cache_key = str(image_path.resolve())
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = (
            "You are observing the target image for an image-retrieval user simulator. "
            "Return only visible, useful retrieval details. Do not infer hidden facts. "
            "Prefer concrete objects, attributes, colors, scene, actions, and spatial relations. "
            "Return JSON only with this schema: "
            "{\"caption\":\"short caption\",\"visual_facts\":[\"fact\",...],"
            "\"uncertain_facts\":[\"fact\",...]}"
        )
        if self.provider == "openai":
            answer = self._openai_service().generate_answer_with_image_base64(
                user_prompt=prompt,
                image_base64_data_url=self._image_to_data_url(image_path),
                model=self.model_name,
                temperature=0.0,
                max_output_tokens=self.max_output_tokens,
                store=False,
            )
        else:
            answer = self._generate_local_observation(image_path=image_path, prompt=prompt)
        try:
            data = self._safe_json_loads(answer)
        except Exception:
            data = self._observation_from_text(answer)
        observation = {
            "provider": self.provider,
            "model": self.model_name,
            "image_file": str(image_path),
            "caption": str(data.get("caption") or "").strip(),
            "visual_facts": self._clean_list(data.get("visual_facts"), limit=12),
            "uncertain_facts": self._clean_list(data.get("uncertain_facts"), limit=8),
        }
        self._cache[cache_key] = observation
        return observation
