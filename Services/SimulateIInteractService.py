import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Database.db_session import SessionLocal
from Entities.entities import VisDialTargetAnnotations
from Experiments.e1_baseline_retrieval import (
    build_rewrite_context,
    compute_rank,
    resolve_torch_device,
)
from Experiments.e2_rair_comparison import score_oracle_suggestions
from Experiments.e3_turn_by_turn import (
    compose_query_from_interaction_state,
    initialize_interaction_state,
    update_interaction_state,
)
from Services.LocalLLMService import LocalLLMService
from Services.OpenAIService import OpenAIService
from Services.TargetObservationService import TargetObservationService
from Services.UserSimulationService import UserSimulationService
from Services.VLMUserSimulationService import VLMUserSimulationService
from Services.VisDialGPTCLIPService import VisDialGPTCLIPService


logger = logging.getLogger("rair.simulate_interact")


class SimulateIInteractService:
    """
    Terminal-only simulator for inspecting one RAIR-VF interaction session.

    It samples a target from VisDialTargetAnnotations, runs retrieval + QVFS +
    RAIR suggestions + target-aware user simulation, and prints the interaction
    as a compact chat transcript.
    """

    def __init__(
        self,
        service: VisDialGPTCLIPService,
        user_simulator: UserSimulationService,
        target_observer: Optional[TargetObservationService] = None,
        split: str = "train",
        evidence_top_k: int = 10,
        fact_top_m: int = 4,
        search_depth: int = 100,
    ) -> None:
        self.service = service
        self.user_simulator = user_simulator
        self.target_observer = target_observer
        self.split = split
        self.evidence_top_k = evidence_top_k
        self.fact_top_m = fact_top_m
        self.search_depth = search_depth

    @staticmethod
    def _ensure_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item or "").strip()]
        return [str(value).strip()] if str(value or "").strip() else []

    @classmethod
    def _row_to_sample(cls, row: VisDialTargetAnnotations) -> Dict[str, Any]:
        return {
            "split": row.split,
            "dialog_index": row.dialog_index,
            "image_id": row.image_id,
            "image_path": row.image_path,
            "base_caption": row.base_caption,
            "enriched_caption": row.enriched_caption,
            "visual_facts": cls._ensure_list(row.visual_facts),
            "positive_facts": cls._ensure_list(row.positive_facts),
            "negative_facts": cls._ensure_list(row.negative_facts),
            "uncertain_facts": cls._ensure_list(row.uncertain_facts),
        }

    def load_target(
        self,
        image_id: Optional[str] = None,
        dialog_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        with SessionLocal() as session:
            stmt = select(VisDialTargetAnnotations).where(
                VisDialTargetAnnotations.split == self.split
            )
            if image_id:
                stmt = stmt.where(VisDialTargetAnnotations.image_id == str(image_id))
            if dialog_index is not None:
                stmt = stmt.where(VisDialTargetAnnotations.dialog_index == dialog_index)

            if not image_id and dialog_index is None:
                stmt = stmt.order_by(func.random())
            else:
                stmt = stmt.order_by(VisDialTargetAnnotations.dialog_index)

            row = session.execute(stmt.limit(1)).scalars().first()

        if row is None:
            raise ValueError(
                f"No VisDialTargetAnnotations target found split={self.split} "
                f"image_id={image_id} dialog_index={dialog_index}"
            )
        return self._row_to_sample(row)

    @staticmethod
    def _print_separator(title: str = "") -> None:
        line = "=" * 90
        print(f"\n{line}")
        if title:
            print(title)
            print(line)

    @staticmethod
    def _print_list(title: str, items: List[Any], limit: int = 8) -> None:
        print(f"{title}:")
        shown = items[:limit]
        if not shown:
            print("  - (none)")
            return
        for item in shown:
            print(f"  - {item}")

    @staticmethod
    def _short_json(data: Any) -> str:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    def _print_target(self, sample: Dict[str, Any]) -> None:
        self._print_separator("TARGET")
        print(f"split/dialog: {sample.get('split')} / {sample.get('dialog_index')}")
        print(f"image_id:     {sample.get('image_id')}")
        print(f"image_path:   {sample.get('image_path')}")
        print(f"caption:      {sample.get('base_caption')}")
        self._print_list("target visual facts", sample.get("visual_facts") or [], limit=10)
        observation = sample.get("target_observation") or {}
        if observation:
            if observation.get("error"):
                print(f"target image observation: {observation.get('error')}")
            else:
                print(f"image observation caption: {observation.get('caption')}")
                self._print_list(
                    "image observation facts",
                    observation.get("visual_facts") or [],
                    limit=10,
                )

    def _print_retrieval(self, rank: Optional[int], image_ids: List[Any], captions: List[str]) -> None:
        print(f"\nRAIR RETRIEVAL: target_rank={rank if rank is not None else 'not found'}")
        for idx, (image_id, caption) in enumerate(zip(image_ids[:5], captions[:5]), start=1):
            print(f"  #{idx:02d} {image_id} | {caption}")

    def _print_evidence(self, evidence: List[Dict[str, Any]]) -> None:
        print("\nRAIR QVFS EVIDENCE:")
        for item in evidence[: self.evidence_top_k]:
            facts = item.get("visual_facts") or []
            print(f"  cand#{item.get('rank')} {item.get('image_id')} | {item.get('caption')}")
            for fact in facts[: self.fact_top_m]:
                print(f"    - {fact}")

    async def run(
        self,
        turns: int = 3,
        image_id: Optional[str] = None,
        dialog_index: Optional[int] = None,
        stop_on_hit_k: Optional[int] = None,
    ) -> None:
        sample = self.load_target(image_id=image_id, dialog_index=dialog_index)
        if self.target_observer is not None:
            sample["target_observation"] = self.target_observer.observe(sample)
        self._print_target(sample)

        print("\nUSER: I want to find this target image.")
        print(f"USER INITIAL CAPTION: {sample.get('base_caption')}")

        gallery = self.service.build_gallery()
        current_query, rewrite_ms = await self.service.rewrite_query(build_rewrite_context(sample))
        interaction_state = initialize_interaction_state(current_query)
        print(f"\nRAIR REWRITE ({rewrite_ms} ms): {current_query}")
        print(f"RAIR MEMORY: {self._short_json(interaction_state)}")

        current_ids, current_captions, current_scores = self.service.faiss_search(
            query_text=current_query,
            gallery=gallery,
            top_k=self.search_depth,
        )
        current_rank = compute_rank(current_ids, sample)
        self._print_retrieval(current_rank, current_ids, current_captions)

        for turn_id in range(1, turns + 1):
            self._print_separator(f"TURN {turn_id}")
            if stop_on_hit_k is not None and current_rank is not None and current_rank <= stop_on_hit_k:
                print(
                    f"USER: I can see the target within top {stop_on_hit_k}; "
                    "I stop refining and select it."
                )
                self._print_retrieval(current_rank, current_ids, current_captions)
                continue

            evidence_raw = self.service.build_candidate_evidence(
                image_ids=current_ids,
                captions=current_captions,
                scores=current_scores,
                top_k=self.evidence_top_k,
            )
            evidence = self.service.select_candidate_facts(
                query=current_query,
                candidate_evidence=evidence_raw,
            )
            self._print_evidence(evidence)

            reasoning = await self.service.reasoning(
                history=None,
                input=current_query,
                candidate_evidence=evidence,
            )
            suggestions = score_oracle_suggestions(
                reasoning.get("suggestions", []),
                sample,
                current_query=current_query,
            )

            print("\nRAIR DIAGNOSIS:")
            print(self._short_json(reasoning.get("diagnosis", {})))
            print("\nRAIR SUGGESTIONS:")
            if suggestions:
                for idx, suggestion in enumerate(suggestions, start=1):
                    print(
                        f"  {idx}. {suggestion.get('sug')} "
                        f"[{suggestion.get('type')}] - {suggestion.get('explain')}"
                    )
            else:
                print("  - (no usable suggestions)")

            user_feedback, user_ms = await self.user_simulator.simulate_turn(
                sample=sample,
                current_query=current_query,
                suggestions=suggestions,
                candidate_evidence=evidence,
                turn_id=turn_id,
                interaction_state=interaction_state,
            )
            print(f"\nSIMULATED USER ({user_ms} ms):")
            print(f"  action: {user_feedback.get('action')}")
            print(f"  reason: {user_feedback.get('reason')}")
            self._print_list("  kept", user_feedback.get("kept_constraints") or [], limit=6)
            self._print_list("  added", user_feedback.get("added_constraints") or [], limit=6)
            self._print_list("  rejected", user_feedback.get("rejected_constraints") or [], limit=6)

            if user_feedback.get("action") == "reject":
                next_query = current_query
                print("\nRAIR MEMORY: unchanged")
            else:
                interaction_state = update_interaction_state(
                    state=interaction_state,
                    user_simulation=user_feedback,
                    fallback_query=current_query,
                )
                rule_query = compose_query_from_interaction_state(
                    state=interaction_state,
                    fallback_query=user_feedback.get("refined_query") or current_query,
                )
                next_query, compose_ms = await self.service.compose_query_from_state(
                    interaction_state=interaction_state,
                    fallback_query=rule_query,
                )
                print("\nRAIR MEMORY:")
                print(self._short_json(interaction_state))
                print(f"RAIR RULE QUERY: {rule_query}")
                print(f"RAIR REWRITE ({compose_ms} ms): {next_query}")

            current_query = next_query
            current_ids, current_captions, current_scores = self.service.faiss_search(
                query_text=current_query,
                gallery=gallery,
                top_k=self.search_depth,
            )
            current_rank = compute_rank(current_ids, sample)
            self._print_retrieval(current_rank, current_ids, current_captions)


def build_llm_service(args: argparse.Namespace) -> Any:
    if args.llm_provider == "local":
        return LocalLLMService(
            model_name=args.local_llm_model,
            device=args.local_llm_device,
            dtype=args.local_llm_dtype,
            local_files_only=not args.local_llm_online,
            max_new_tokens=args.local_llm_max_new_tokens,
        )
    return OpenAIService(model_name=args.reasoning_model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print one terminal chat between RAIR and a target-aware user simulator."
    )
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--image-id", default=None)
    parser.add_argument("--dialog-index", type=int, default=None)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--stop-on-hit-k", type=int, default=None)
    parser.add_argument("--search-depth", type=int, default=100)
    parser.add_argument("--evidence-top-k", type=int, default=10)
    parser.add_argument("--fact-top-m", type=int, default=4)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=resolve_torch_device("CLIP_DEVICE"))
    parser.add_argument("--llm-provider", choices=["openai", "local"], default=os.environ.get("RAIR_LLM_PROVIDER", "local"))
    parser.add_argument("--reasoning-model", default=os.environ.get("RAIR_REASONING_MODEL", "gpt-4o"))
    parser.add_argument("--local-llm-model", default=os.environ.get("RAIR_LOCAL_LLM_MODEL", "google/gemma-3-12b-it"))
    parser.add_argument("--local-llm-device", default=resolve_torch_device("LOCAL_LLM_DEVICE"))
    parser.add_argument("--local-llm-dtype", default=os.environ.get("LOCAL_LLM_DTYPE", "bfloat16"))
    parser.add_argument("--local-llm-max-new-tokens", type=int, default=768)
    parser.add_argument("--local-llm-online", action="store_true")
    parser.add_argument("--user-sim-provider", choices=["text", "vlm"], default="text")
    parser.add_argument("--user-sim-model", default=None)
    parser.add_argument("--user-sim-max-output-tokens", type=int, default=512)
    parser.add_argument("--user-sim-temperature", type=float, default=0.0)
    parser.add_argument("--user-sim-vlm-model", default=os.environ.get("RAIR_USER_SIM_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--user-sim-vlm-device", default=os.environ.get("RAIR_USER_SIM_VLM_DEVICE", "cuda"))
    parser.add_argument("--user-sim-vlm-dtype", default=os.environ.get("RAIR_USER_SIM_VLM_DTYPE", "bfloat16"))
    parser.add_argument("--user-sim-vlm-online", action="store_true")
    parser.add_argument("--target-observation-provider", choices=["none", "openai", "local"], default="none")
    parser.add_argument("--target-observation-model", default=os.environ.get("RAIR_TARGET_OBSERVATION_MODEL"))
    parser.add_argument("--target-observation-image-root", default=os.environ.get("RAIR_IMAGE_ROOT"))
    parser.add_argument("--target-observation-max-output-tokens", type=int, default=512)
    parser.add_argument("--target-observation-device", default=os.environ.get("RAIR_TARGET_OBSERVATION_DEVICE", "cuda"))
    parser.add_argument("--target-observation-dtype", default=os.environ.get("RAIR_TARGET_OBSERVATION_DTYPE", "bfloat16"))
    parser.add_argument("--target-observation-online", action="store_true")
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    llm_service = build_llm_service(args)
    service = VisDialGPTCLIPService(
        vlm=args.clip_model,
        device=args.device,
        openai_service=llm_service,
        reasoning_model=args.local_llm_model if args.llm_provider == "local" else args.reasoning_model,
        evidence_top_k=args.evidence_top_k,
        fact_top_m=args.fact_top_m,
    )
    if args.user_sim_provider == "vlm":
        user_sim_vlm = TargetObservationService(
            provider="local",
            model_name=args.user_sim_vlm_model,
            image_root=args.target_observation_image_root,
            max_output_tokens=args.user_sim_max_output_tokens,
            device=args.user_sim_vlm_device,
            dtype=args.user_sim_vlm_dtype,
            local_files_only=not args.user_sim_vlm_online,
        )
        simulator = VLMUserSimulationService(
            target_vlm=user_sim_vlm,
            max_output_tokens=args.user_sim_max_output_tokens,
            temperature=args.user_sim_temperature,
        )
    else:
        simulator = UserSimulationService(
            llm_service=llm_service,
            model_name=args.user_sim_model,
            max_output_tokens=args.user_sim_max_output_tokens,
            temperature=args.user_sim_temperature,
        )
    target_observer = None
    if args.target_observation_provider != "none":
        observation_model = args.target_observation_model
        if not observation_model:
            observation_model = (
                args.local_llm_model
                if args.target_observation_provider == "local"
                else args.reasoning_model
            )
        target_observer = TargetObservationService(
            provider=args.target_observation_provider,
            model_name=observation_model,
            image_root=args.target_observation_image_root,
            max_output_tokens=args.target_observation_max_output_tokens,
            device=args.target_observation_device,
            dtype=args.target_observation_dtype,
            local_files_only=not args.target_observation_online,
        )
    runner = SimulateIInteractService(
        service=service,
        user_simulator=simulator,
        target_observer=target_observer,
        split=args.split,
        evidence_top_k=args.evidence_top_k,
        fact_top_m=args.fact_top_m,
        search_depth=args.search_depth,
    )
    await runner.run(
        turns=args.turns,
        image_id=args.image_id,
        dialog_index=args.dialog_index,
        stop_on_hit_k=args.stop_on_hit_k,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
