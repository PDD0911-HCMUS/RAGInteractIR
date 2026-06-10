import re
from typing import Any, Dict, Iterable, List, Optional


class QAFS:
    """
    Query-Aware Visual Fact Selection.

    Scores candidate visual facts with:
      score(f | q, c_i) =
          alpha * sim_clip_text(q, f)
        + beta  * lexical_overlap(q, f)
        + gamma * discriminativeness(f, candidates)
        - delta * contradiction_penalty(f, q)

    The service expects an embedding service with embed_texts(texts), which is
    already provided by VisDialGPTCLIPService.
    """

    DEFAULT_STOPWORDS = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "with",
        "is", "are", "was", "were", "be", "being", "been", "this", "that",
        "there", "it", "he", "she", "they", "them", "his", "her", "their",
        "image", "photo", "picture", "visual", "facts", "additional",
    }

    DEFAULT_CONTRADICTION_PAIRS = [
        ("inside", "outside"),
        ("indoor", "outdoor"),
        ("indoors", "outdoors"),
        ("day", "night"),
        ("black", "white"),
        ("standing", "sitting"),
        ("sitting", "lying"),
        ("laying", "standing"),
        ("grass", "floor"),
    ]

    def __init__(
        self,
        embedding_service: Any,
        top_m: int = 4,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        delta: float = 0.5,
        stopwords: Optional[set[str]] = None,
        contradiction_pairs: Optional[List[tuple[str, str]]] = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.top_m = top_m
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.stopwords = stopwords or self.DEFAULT_STOPWORDS
        self.contradiction_pairs = contradiction_pairs or self.DEFAULT_CONTRADICTION_PAIRS

    def tokenize(self, text: Any) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
            if len(token) > 2 and token not in self.stopwords
        }

    @staticmethod
    def dedupe_keep_order(items: Iterable[Any]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            clean = " ".join(str(item or "").strip().split())
            key = clean.lower()
            if not clean or key in seen:
                continue
            seen.add(key)
            result.append(clean)
        return result

    def evidence_facts(self, candidate: Dict[str, Any]) -> List[str]:
        return self.dedupe_keep_order(
            [
                *(candidate.get("visual_facts") or []),
                *(candidate.get("positive_facts") or []),
                *(candidate.get("negative_facts") or []),
                *(candidate.get("uncertain_facts") or []),
            ]
        )

    @staticmethod
    def lexical_overlap_score(query_tokens: set[str], fact_tokens: set[str]) -> float:
        if not fact_tokens:
            return 0.0
        return len(query_tokens & fact_tokens) / len(fact_tokens)

    def contradiction_penalty(self, query_tokens: set[str], fact_tokens: set[str]) -> float:
        penalty = 0.0
        for left, right in self.contradiction_pairs:
            if left in query_tokens and right in fact_tokens:
                penalty += 1.0
            if right in query_tokens and left in fact_tokens:
                penalty += 1.0

        negation_tokens = {"no", "not", "without"}
        if query_tokens & fact_tokens and fact_tokens & negation_tokens:
            penalty += 0.5
        return penalty

    @staticmethod
    def discriminativeness_score(
        fact_tokens: set[str],
        candidate_token_df: Dict[str, int],
        num_candidates: int,
    ) -> float:
        if not fact_tokens or num_candidates <= 1:
            return 0.0

        scores = []
        for token in fact_tokens:
            df = candidate_token_df.get(token, 1)
            scores.append(1.0 - (df - 1) / max(1, num_candidates - 1))
        return sum(scores) / len(scores)

    def sim_clip_text(self, query: str, facts: List[str]) -> Dict[str, float]:
        if not facts:
            return {}

        embeddings = self.embedding_service.embed_texts([query, *facts])
        query_embedding = embeddings[0]
        fact_embeddings = embeddings[1:]
        similarities = (fact_embeddings @ query_embedding).tolist()
        return {
            fact: max(0.0, min(1.0, (float(score) + 1.0) / 2.0))
            for fact, score in zip(facts, similarities)
        }

    def candidate_token_document_frequency(
        self,
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        token_df: Dict[str, int] = {}
        for candidate in evidence:
            candidate_tokens = set()
            for fact in self.evidence_facts(candidate):
                candidate_tokens.update(self.tokenize(fact))
            for token in candidate_tokens:
                token_df[token] = token_df.get(token, 0) + 1
        return token_df

    def score_fact(
        self,
        query_tokens: set[str],
        fact: str,
        clip_scores: Dict[str, float],
        candidate_token_df: Dict[str, int],
        num_candidates: int,
    ) -> Dict[str, Any]:
        fact_tokens = self.tokenize(fact)
        components = {
            "clip": clip_scores.get(fact, 0.0),
            "lexical": self.lexical_overlap_score(query_tokens, fact_tokens),
            "discriminative": self.discriminativeness_score(
                fact_tokens=fact_tokens,
                candidate_token_df=candidate_token_df,
                num_candidates=num_candidates,
            ),
            "contradiction": self.contradiction_penalty(query_tokens, fact_tokens),
        }
        score = (
            self.alpha * components["clip"]
            + self.beta * components["lexical"]
            + self.gamma * components["discriminative"]
            - self.delta * components["contradiction"]
        )
        return {
            "fact": fact,
            "score": score,
            "components": components,
        }

    def select(self, query: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.top_m <= 0:
            return evidence

        all_facts = self.dedupe_keep_order(
            fact
            for candidate in evidence
            for fact in self.evidence_facts(candidate)
        )
        clip_scores = self.sim_clip_text(query, all_facts)
        query_tokens = self.tokenize(query)
        num_candidates = len(evidence)
        candidate_token_df = self.candidate_token_document_frequency(evidence)

        selected_evidence = []
        for candidate in evidence:
            facts = self.evidence_facts(candidate)
            scored_facts = [
                self.score_fact(
                    query_tokens=query_tokens,
                    fact=fact,
                    clip_scores=clip_scores,
                    candidate_token_df=candidate_token_df,
                    num_candidates=num_candidates,
                )
                for fact in facts
            ]
            scored_facts.sort(key=lambda item: item["score"], reverse=True)

            selected_facts = [item["fact"] for item in scored_facts[: self.top_m]]
            selected_set = set(selected_facts)

            item = dict(candidate)
            item["visual_facts"] = selected_facts
            item["positive_facts"] = [
                fact for fact in candidate.get("positive_facts", []) if fact in selected_set
            ]
            item["negative_facts"] = [
                fact for fact in candidate.get("negative_facts", []) if fact in selected_set
            ]
            item["uncertain_facts"] = [
                fact for fact in candidate.get("uncertain_facts", []) if fact in selected_set
            ]
            item["enriched_caption"] = (
                f"{candidate.get('caption', '')}. Selected visual facts: "
                + "; ".join(selected_facts)
                if selected_facts
                else candidate.get("caption", "")
            )
            item["qafs"] = {
                "original_fact_count": len(facts),
                "selected_fact_count": len(selected_facts),
                "selected": scored_facts[: self.top_m],
                "weights": {
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "gamma": self.gamma,
                    "delta": self.delta,
                },
            }
            selected_evidence.append(item)

        return selected_evidence


# Backward-compatible alias for earlier experiment imports.
QASF = QAFS
