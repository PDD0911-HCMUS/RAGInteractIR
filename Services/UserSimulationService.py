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

    @classmethod
    def _clean_list(cls, values: Any, limit: int = 12) -> List[str]:
        if not isinstance(values, list):
            return []
        cleaned: List[str] = []
        seen = set()
        for value in values:
            text = cls._sanitize_query(str(value or ""))
            key = text.lower()
            if text and key not in seen:
                cleaned.append(text)
                seen.add(key)
            if len(cleaned) >= limit:
                break
        return cleaned

    @staticmethod
    def _extract_query_constraints(query: str) -> Dict[str, List[str]]:
        """
        Lightweight query parsing for the simulator prompt. This is not used as
        a hard rule; it gives the LLM explicit polarity hints so terms such as
        "-drawing" or "not indoor" are treated as exclusions.
        """
        text = str(query or "").strip()
        negative: List[str] = []

        for match in re.finditer(r"(?<!\w)-([A-Za-z][A-Za-z0-9_-]*)", text):
            negative.append(match.group(1).replace("_", " ").replace("-", " ").strip())

        for match in re.finditer(
            r"\b(?:not|no|without|exclude|excluding)\s+([A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z][A-Za-z0-9_-]*){0,2})",
            text,
            flags=re.IGNORECASE,
        ):
            negative.append(match.group(1).strip(" ,.;:"))

        positive_text = re.sub(r"(?<!\w)-[A-Za-z][A-Za-z0-9_-]*", " ", text)
        positive_text = re.sub(
            r"\b(?:not|no|without|exclude|excluding)\s+[A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z][A-Za-z0-9_-]*){0,2}",
            " ",
            positive_text,
            flags=re.IGNORECASE,
        )
        positive = [
            chunk.strip(" ,.;:")
            for chunk in re.split(r"\s*[;,]\s*", positive_text)
            if chunk.strip(" ,.;:")
        ]

        deduped_negative = []
        seen = set()
        for item in negative:
            key = item.lower()
            if item and key not in seen:
                seen.add(key)
                deduped_negative.append(item)

        return {
            "positive_text": positive[:6],
            "negative_text": deduped_negative[:8],
            "note": "Negative constraints are exclusions/absence, not positive visual details.",
        }

    def _build_context(
        self,
        sample: Dict[str, Any],
        current_query: str,
        suggestions: List[Dict[str, Any]],
        candidate_evidence: List[Dict[str, Any]],
        turn_id: int,
        interaction_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = interaction_state or {}
        active_text = " ".join(
            [
                current_query,
                " ".join(state.get("positive_constraints") or []),
                " ".join(state.get("negative_constraints") or []),
            ]
        ).lower()
        target_observation = sample.get("target_observation") or {}
        observed_facts = self._clean_list(
            target_observation.get("visual_facts") or [],
            limit=12,
        )
        target_positive = self._fact_list(sample, "positive_facts")[:12]
        hybrid_positive = self._clean_list(observed_facts + target_positive, limit=16)
        missing_target_facts = [
            fact for fact in hybrid_positive if fact.lower() not in active_text
        ][:6]
        return {
            "turn": turn_id,
            "current_query": current_query,
            "current_query_constraints": self._extract_query_constraints(current_query),
            "interaction_state": state,
            "missing_target_facts": missing_target_facts,
            "target": {
                "image_id": sample.get("image_id"),
                "image_path": sample.get("image_path"),
                "caption": sample.get("base_caption", ""),
                "image_observation": {
                    "caption": target_observation.get("caption", ""),
                    "visual_facts": observed_facts,
                    "uncertain_facts": self._clean_list(
                        target_observation.get("uncertain_facts") or [],
                        limit=8,
                    ),
                },
                "visual_facts": self._fact_list(sample, "visual_facts")[:12],
                "positive_facts": hybrid_positive,
                "annotation_positive_facts": target_positive,
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
        interaction_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], int]:
        context = self._build_context(
            sample=sample,
            current_query=current_query,
            suggestions=suggestions,
            candidate_evidence=candidate_evidence,
            turn_id=turn_id,
            interaction_state=interaction_state,
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
        if action == "reject":
            refined_query = current_query

        result = {
            "action": action,
            "selected_suggestions": data.get("selected_suggestions") or [],
            "kept_constraints": self._clean_list(data.get("kept_constraints") or [], limit=16),
            "added_constraints": self._clean_list(data.get("added_constraints") or [], limit=12),
            "negative_constraints": self._clean_list(data.get("negative_constraints") or [], limit=12),
            "rejected_constraints": self._clean_list(data.get("rejected_constraints") or [], limit=12),
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
