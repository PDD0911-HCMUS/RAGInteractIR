from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field


# -----------------------
# 1) Output schema (Pydantic for validation only)
# -----------------------
class Triplet(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: Optional[float] = None


class TripletGraph(BaseModel):
    triplets: List[Triplet]
    meta: Dict[str, Any] = Field(default_factory=dict)


# -----------------------
# 2) Helpers
# -----------------------
def img_to_b64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def extract_json(text: str) -> str:
    """
    Extract the LAST JSON object from model output.
    Handles cases where model prints extra text or ```json blocks.
    """
    # Prefer fenced ```json ... ```
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced[-1].strip()

    # Fallback: find last {...} block (greedy from last '{' to last '}')
    start = text.rfind("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()

    raise ValueError("No JSON object found in model output.")


def build_prompt() -> str:
    # IMPORTANT: do NOT include JSON schema or backslashes like \_
    return """
           You are a vision-language model.

            From the given image, please generate the caption for this image, around 1 or 2 caption. Be factual and do not guess.
            """.strip()


# -----------------------
# 3) Main
# -----------------------
def image_to_triplets(image_path: str) -> TripletGraph:
    llm = ChatOllama(
        model="llava-llama3",
        temperature=0.07,
        base_url="http://127.0.0.1:11434",
        # important for stability:
        stream=False,
        keep_alive=0,
    )

    msg = HumanMessage(content=build_prompt(), images=[image_path])

    raw = llm.invoke([msg])
    text = raw.content

    return text


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dialog/qna.py <image_path>")
        raise SystemExit(1)

    image_path = sys.argv[1]
    tg = image_to_triplets(image_path)

    print(tg)
