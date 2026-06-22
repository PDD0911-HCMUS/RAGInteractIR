import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from Services.PromptCollectionService import PromptCollectionService


logger = logging.getLogger("rair.user_sim")


class UserSimulationService:
    """
    Target-aware user simulator for interactive retrieval experiments.

    The RAIR system never sees target facts directly. This simulator is used
    only by the experiment harness to emulate how a user who knows the target
    image would respond to RAIR suggestions.
    """

    def __init__(
        self,
        llm_service: Any,
        model_name: Optional[str] = None,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.llm_service = llm_service
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.prompt = PromptCollectionService()

    @staticmethod
    def _compact_json(data: Any) -> str:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _safe_json_loads(text: str) -> Any:
        if not isinstance(text, str):
            return text

        cleaned = text.strip().removeprefix("\ufeff").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except Exception as direct_error:
            decoder = json.JSONDecoder()
            starts = [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx >= 0]
            for start in sorted(starts):
                try:
                    value, _ = decoder.raw_decode(cleaned[start:])
                    return value
                except Exception:
                    continue
            raise ValueError(f"Invalid JSON returned by user simulator: {text!r}") from direct_error

    @staticmethod
    def _fact_list(sample: Dict[str, Any], key: str) -> List[str]:
        values = sample.get(key) or []
        return [str(value).strip() for value in values if str(value or "").strip()]

    @staticmethod
    def _compact_evidence(evidence: List[Dict[str, Any]], max_candidates: int = 3) -> List[Dict[str, Any]]:
        compact = []
        for item in evidence[:max_candidates]:
            compact.append(
                {
                    "rank": item.get("rank"),
                    "caption": item.get("caption"),
                    "visual_facts": item.get("visual_facts", [])[:5],
                    "positive_facts": item.get("positive_facts", [])[:5],
                    "negative_facts": item.get("negative_facts", [])[:3],
                    "uncertain_facts": item.get("uncertain_facts", [])[:3],
                }
            )
        return compact

    @staticmethod
    def _compact_suggestions(suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact = []
        for item in suggestions:
            compact.append(
                {
                    "sug": item.get("sug", ""),
                    "type": item.get("type", ""),
                    "explain": item.get("explain", ""),
                    "target_overlap": item.get("oracle_overlap"),
                    "novel_target_overlap": item.get("oracle_novel_overlap"),
                }
            )
        return compact

    @staticmethod
    def _normalize_action(action: str) -> str:
        action = str(action or "").strip().lower()
        allowed = {
            "accept",
            "edit",
            "combine",
            "reject",
            "add_detail",
            "remove_detail",
        }
        return action if action in allowed else "reject"

    @staticmethod
    def _sanitize_query(query: str) -> str:
        query = str(query or "").strip()
        query = re.sub(r"\s+", " ", query)
        query = query.strip(" ;,.")
        return query

    def _build_context(
        self,
        sample: Dict[str, Any],
        current_query: str,
        suggestions: List[Dict[str, Any]],
        candidate_evidence: List[Dict[str, Any]],
        turn_id: int,
    ) -> Dict[str, Any]:
        return {
            "turn": turn_id,
            "current_query": current_query,
            "target": {
                "image_id": sample.get("image_id"),
                "image_path": sample.get("image_path"),
                "caption": sample.get("base_caption", ""),
                "visual_facts": self._fact_list(sample, "visual_facts")[:12],
                "positive_facts": self._fact_list(sample, "positive_facts")[:12],
                "negative_facts": self._fact_list(sample, "negative_facts")[:8],
                "uncertain_facts": self._fact_list(sample, "uncertain_facts")[:8],
            },
            "system_suggestions": self._compact_suggestions(suggestions),
            "retrieved_candidate_evidence": self._compact_evidence(candidate_evidence),
            "policy": {
                "accept": "use a good suggestion as-is when it is target-supported",
                "edit": "rewrite a partially useful suggestion to match target facts",
                "combine": "merge multiple useful suggestions",
                "reject": "keep the current query when suggestions are unsupported",
                "add_detail": "add a target fact missing from suggestions if helpful",
                "remove_detail": "remove a query detail that conflicts with target facts",
            },
        }

    async def simulate_turn(
        self,
        sample: Dict[str, Any],
        current_query: str,
        suggestions: List[Dict[str, Any]],
        candidate_evidence: List[Dict[str, Any]],
        turn_id: int,
    ) -> Tuple[Dict[str, Any], int]:
        context = self._build_context(
            sample=sample,
            current_query=current_query,
            suggestions=suggestions,
            candidate_evidence=candidate_evidence,
            turn_id=turn_id,
        )
        prompt = self.prompt.simulate_user_edit.format(context=self._compact_json(context))

        start = time.time()
        answer = self.llm_service.generate_answer(
            user_prompt=prompt,
            history=None,
            model=self.model_name,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            store=False,
        )
        latency_ms = int((time.time() - start) * 1000)

        data = self._safe_json_loads(answer)
        if not isinstance(data, dict):
            raise ValueError(f"User simulator response must be a JSON object: {data!r}")

        action = self._normalize_action(data.get("action", "reject"))
        refined_query = self._sanitize_query(data.get("refined_query", ""))
        if action == "reject" or not refined_query:
            action = "reject"
            refined_query = current_query

        result = {
            "action": action,
            "selected_suggestions": data.get("selected_suggestions") or [],
            "added_target_details": data.get("added_target_details") or [],
            "removed_details": data.get("removed_details") or [],
            "refined_query": refined_query,
            "reason": str(data.get("reason", "")).strip(),
        }
        logger.info(
            "User simulation turn=%s action=%s refined_query=%s",
            turn_id,
            action,
            refined_query,
        )
        return result, latency_ms
