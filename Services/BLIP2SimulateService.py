import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


logger = logging.getLogger("rair")


class BLIP2SimulateService:
    """
    Simulate a target-aware user for interactive retrieval evaluation.

    The simulator sees the ground-truth image and answers the system's
    suggestions as if it were a user trying to help retrieve that image.
    """

    DEFAULT_MODEL = "Salesforce/blip2-flan-t5-xl"

    _processor = None
    _model = None
    _device = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        local_files_only: Optional[bool] = None,
        max_new_tokens: int = 64,
        image_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.model_name = model_name or os.environ.get(
            "BLIP2_MODEL_NAME",
            self.DEFAULT_MODEL,
        )
        self.device = device or os.environ.get("BLIP2_DEVICE")
        self.max_new_tokens = max_new_tokens
        self.image_root = Path(
            image_root
            or os.environ.get(
                "BLIP2_IMAGE_ROOT",
                Path(__file__).resolve().parents[1] / "datasets" / "MSCOCO",
            )
        )

        if local_files_only is None:
            local_files_only = os.environ.get("HF_LOCAL_FILES_ONLY", "1") != "0"
        self.local_files_only = local_files_only

    def _resolve_device(self):
        if self.device:
            return self.device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _get_or_create_model(self):
        if self.__class__._model is None or self.__class__._processor is None:
            with self.__class__._lock:
                if self.__class__._model is None or self.__class__._processor is None:
                    import torch
                    from transformers import (
                        AutoTokenizer,
                        Blip2ForConditionalGeneration,
                        Blip2Processor,
                        BlipImageProcessor,
                    )

                    device = self._resolve_device()
                    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

                    logger.info(
                        "Loading BLIP-2 simulator model=%s device=%s local_files_only=%s",
                        self.model_name,
                        device,
                        self.local_files_only,
                    )

                    try:
                        try:
                            processor = Blip2Processor.from_pretrained(
                                self.model_name,
                                local_files_only=self.local_files_only,
                            )
                        except TypeError as exc:
                            if "num_query_tokens" not in str(exc):
                                raise
                            logger.warning(
                                "Blip2Processor config has num_query_tokens unsupported by "
                                "this transformers version; loading processor components manually."
                            )
                            image_processor = BlipImageProcessor.from_pretrained(
                                self.model_name,
                                local_files_only=self.local_files_only,
                            )
                            tokenizer = AutoTokenizer.from_pretrained(
                                self.model_name,
                                local_files_only=self.local_files_only,
                            )
                            processor = Blip2Processor(
                                image_processor=image_processor,
                                tokenizer=tokenizer,
                            )

                        model = Blip2ForConditionalGeneration.from_pretrained(
                            self.model_name,
                            torch_dtype=dtype,
                            local_files_only=self.local_files_only,
                        )
                    except OSError as exc:
                        cache_hint = (
                            "BLIP-2 model is not available in the local Hugging Face cache. "
                            "Run evaluation with `--no-hf-local-files-only` to download it, "
                            "or set BLIP2_MODEL_NAME to a local model path."
                        )
                        raise RuntimeError(cache_hint) from exc
                    model.to(device)
                    model.eval()

                    self.__class__._processor = processor
                    self.__class__._model = model
                    self.__class__._device = device
                    self.device = device

                    logger.info("BLIP-2 simulator ready")

        if self.device is None and self.__class__._device is not None:
            self.device = self.__class__._device

        return self.__class__._processor, self.__class__._model

    def resolve_image_path(self, image_path: Union[str, Path]) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        return self.image_root / path

    def _load_image(self, image_path: Union[str, Path]):
        from PIL import Image

        path = self.resolve_image_path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Target image not found: {path}")

        return Image.open(path).convert("RGB")

    @staticmethod
    def _compact_suggestions(suggestions: Optional[List[Union[str, Dict[str, Any]]]]) -> List[str]:
        compact: List[str] = []
        for item in suggestions or []:
            if isinstance(item, dict):
                text = str(item.get("sug", "")).strip()
            else:
                text = str(item or "").strip()

            if text:
                compact.append(text)

        return compact

    @staticmethod
    def _context_summary(context_state: Optional[Dict[str, Any]]) -> str:
        if not context_state:
            return "No previous context."

        initial_query = context_state.get("initial_query") or ""
        feedback_pairs = context_state.get("feedback_pairs") or []
        latest_user_message = context_state.get("latest_user_message") or ""

        lines = []
        if initial_query:
            lines.append(f"Initial query: {initial_query}")
        if latest_user_message:
            lines.append(f"Latest user message: {latest_user_message}")

        for pair in feedback_pairs[-3:]:
            pair_suggestions = BLIP2SimulateService._compact_suggestions(
                pair.get("suggestions") or []
            )
            answer = pair.get("answer") or ""
            if pair_suggestions or answer:
                lines.append(
                    "Previous feedback: suggestions="
                    f"{pair_suggestions}; answer={answer}"
                )

        return "\n".join(lines) if lines else "No previous context."

    def build_prompt(
        self,
        suggestions: Optional[List[Union[str, Dict[str, Any]]]] = None,
        context_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        compact_suggestions = self._compact_suggestions(suggestions)
        suggestion_text = (
            "\n".join(f"- {sug}" for sug in compact_suggestions)
            if compact_suggestions
            else "- No explicit suggestion; respond to the latest query."
        )
        context_text = self._context_summary(context_state)

        return (
            "You are simulating a user in an interactive image retrieval task.\n"
            "Look at the target image and answer the system's suggestions so that "
            "the system can retrieve this exact image.\n\n"
            f"Conversation context:\n{context_text}\n\n"
            f"System suggestions:\n{suggestion_text}\n\n"
            "Answer as the user in one concise sentence. "
            "Accept details that match the image, reject wrong details, and add one "
            "important visual correction if needed. Do not mention that you are an AI "
            "or that you are looking at an image."
        )

    def simulate_answer(
        self,
        target_image_path: Union[str, Path],
        suggestions: Optional[List[Union[str, Dict[str, Any]]]] = None,
        context_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        processor, model = self._get_or_create_model()
        image = self._load_image(target_image_path)
        prompt = self.build_prompt(
            suggestions=suggestions,
            context_state=context_state,
        )

        import torch

        device = self._resolve_device()
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        answer = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()
        logger.info("BLIP-2 simulated answer: %s", answer)
        return answer

    def simulate_feedback(
        self,
        target_image_path: Union[str, Path],
        suggestions: Optional[List[Union[str, Dict[str, Any]]]] = None,
        context_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        answer = self.simulate_answer(
            target_image_path=target_image_path,
            suggestions=suggestions,
            context_state=context_state,
        )

        return {
            "target_image_path": str(target_image_path),
            "suggestions": self._compact_suggestions(suggestions),
            "answer": answer,
        }
