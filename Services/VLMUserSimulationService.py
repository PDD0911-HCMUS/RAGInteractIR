import time
from typing import Any, Dict, List, Optional, Tuple

from Services.TargetObservationService import TargetObservationService
from Services.UserSimulationService import UserSimulationService


class VLMUserSimulationService(UserSimulationService):
    """
    User simulator backed by a vision-language model.

    It keeps the same simulate_turn() interface as UserSimulationService, but
    the model receives the target image directly together with the RAIR
    suggestions and candidate evidence. This is closer to an interactive user
    who can inspect the target image while refining the query.
    """

    def __init__(
        self,
        target_vlm: TargetObservationService,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(
            llm_service=None,
            model_name=None,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        self.target_vlm = target_vlm

    @staticmethod
    def _compact_vlm_suggestions(suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact = []
        for item in suggestions:
            compact.append(
                {
                    "sug": item.get("sug", ""),
                    "type": item.get("type", ""),
                    "explain": item.get("explain", ""),
                }
            )
        return compact

    def _build_vlm_context(
        self,
        sample: Dict[str, Any],
        current_query: str,
        suggestions: List[Dict[str, Any]],
        candidate_evidence: List[Dict[str, Any]],
        turn_id: int,
        interaction_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "turn": turn_id,
            "current_query": current_query,
            "current_query_constraints": self._extract_query_constraints(current_query),
            "interaction_state": interaction_state or {},
            "target": {
                "image_id": sample.get("image_id"),
                "image_path": sample.get("image_path"),
                "note": "The target image is provided separately. Do not rely on hidden target annotations.",
            },
            "system_suggestions": self._compact_vlm_suggestions(suggestions),
            "retrieved_candidate_evidence": self._compact_evidence(candidate_evidence),
            "policy": {
                "accept": "use a good suggestion as-is when it matches the target image",
                "edit": "rewrite a partially useful suggestion to match the target image",
                "combine": "merge multiple useful suggestions",
                "reject": "keep the current query when suggestions do not match the target image",
                "add_detail": "add one visible target detail missing from suggestions if helpful",
                "remove_detail": "remove a query detail that conflicts with the target image",
            },
        }

    async def generate_initial_query(self, sample: Dict[str, Any]) -> Tuple[str, int]:
        prompt = (
            "You are a user starting an image retrieval session. Look at the target image "
            "and write the first search query you would naturally type to find it. "
            "Use only visible details from the image. Keep it concise, usually 6 to 14 words. "
            "Do not mention that you are looking at an image. Return JSON only: "
            "{\"initial_query\":\"...\"}"
        )
        start = time.time()
        answer = self.target_vlm.generate_with_image(sample=sample, prompt=prompt)
        latency_ms = int((time.time() - start) * 1000)

        try:
            data = self._safe_json_loads(answer)
            query = self._sanitize_query(data.get("initial_query", ""))
        except Exception:
            query = self._sanitize_query(answer)

        if not query:
            query = self._sanitize_query(sample.get("base_caption", ""))
        return query, latency_ms

    async def simulate_turn(
        self,
        sample: Dict[str, Any],
        current_query: str,
        suggestions: List[Dict[str, Any]],
        candidate_evidence: List[Dict[str, Any]],
        turn_id: int,
        interaction_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], int]:
        context = self._build_vlm_context(
            sample=sample,
            current_query=current_query,
            suggestions=suggestions,
            candidate_evidence=candidate_evidence,
            turn_id=turn_id,
            interaction_state=interaction_state,
        )
        prompt = (
            "You are the simulated user in an interactive image retrieval task. "
            "Look at the target image. Use candidate evidence only to understand what "
            "the retrieval system is currently confusing with the target. Candidate "
            "evidence can be wrong for the target. Choose a small useful edit only if it "
            "would help retrieve the target image. If the target already seems easy to find "
            "from the current query, reject/no-op. Return valid JSON only.\n"
            "Context:"
            + self._compact_json(context)
            + "\nJSON schema:"
            "{{"
            "\"action\":\"accept|edit|combine|reject|add_detail|remove_detail\","
            "\"selected_suggestions\":[\"...\"],"
            "\"kept_constraints\":[\"...\"],"
            "\"added_constraints\":[\"...\"],"
            "\"negative_constraints\":[\"...\"],"
            "\"rejected_constraints\":[\"...\"],"
            "\"added_target_details\":[\"...\"],"
            "\"removed_details\":[\"...\"],"
            "\"refined_query\":\"...\","
            "\"reason\":\"...\""
            "}}"
        )

        start = time.time()
        answer = self.target_vlm.generate_with_image(sample=sample, prompt=prompt)
        latency_ms = int((time.time() - start) * 1000)

        data = self._safe_json_loads(answer)
        if not isinstance(data, dict):
            raise ValueError(f"VLM user simulator response must be a JSON object: {data!r}")

        action = self._normalize_action(data.get("action", "reject"))
        refined_query = self._sanitize_query(data.get("refined_query", ""))
        if action == "reject":
            refined_query = current_query

        return {
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
        }, latency_ms
