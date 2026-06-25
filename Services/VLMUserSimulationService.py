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
        prompt = (
            "You are the simulated user in an interactive image retrieval task. "
            "Look at the target image. Use candidate evidence only to understand what "
            "the retrieval system is currently confusing with the target. Candidate "
            "evidence can be wrong for the target. Return valid JSON only.\n"
            + self.prompt.simulate_user_edit.format(context=self._compact_json(context))
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
