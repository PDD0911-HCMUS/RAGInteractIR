import threading
from typing import Optional, List, Tuple, Literal

from lmdeploy import pipeline
from lmdeploy.vl import load_image

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

Role = Literal["user", "assistant", "system"]
RoleHistory = List[Tuple[Role, str]]

# lmdeploy history commonly expects: List[Tuple[user_text, assistant_text]]
PairHistory = List[Tuple[str, str]]

_pipe = None
_lock = threading.Lock()


def get_pipe():
    global _pipe
    if _pipe is None:
        with _lock:
            if _pipe is None:
                _pipe = pipeline(MODEL_NAME)
    return _pipe


def normalize_history_for_lmdeploy(history: Optional[RoleHistory]) -> PairHistory:
    """
    Convert role-based history:
        [("system","..."), ("user","q1"), ("assistant","a1"), ("user","q2"), ...]
    into pair-based history expected by many chat pipelines:
        [("q1","a1"), ("q2","a2"), ...]
    Notes:
    - system messages are ignored here (you already put system prompt in user_prompt)
    - if the last user message has no assistant reply yet, we drop it
    """
    if not history:
        return []

    pairs: PairHistory = []
    pending_user: Optional[str] = None

    for role, content in history:
        if role == "user":
            pending_user = content
        elif role == "assistant":
            if pending_user is not None:
                pairs.append((pending_user, content))
                pending_user = None
        else:
            # role == "system": ignore
            continue

    return pairs


def _resp_to_text(resp):
    if hasattr(resp, "text"):
        return (resp.text or "").strip()
    return str(resp).strip()


def generate_caption(
    media_path: Optional[str],
    user_prompt: str,
    media_type: str = "image",
):
    pipe = get_pipe()

    # currently we only support image input
    if media_path and media_type == "image":
        img = load_image(media_path)
        resp = pipe((user_prompt, img))  # (text, image)
    else:
        resp = pipe(user_prompt)

    return _resp_to_text(resp)


def generate_answer(
    media_path: Optional[str],
    user_prompt: str,
    history: Optional[RoleHistory] = None,
    media_type: str = "image",
):
    pipe = get_pipe()

    hist_pairs = normalize_history_for_lmdeploy(history)

    if media_path and media_type == "image":
        img = load_image(media_path)
        resp = pipe((user_prompt, img), history=hist_pairs)
    else:
        resp = pipe(user_prompt, history=hist_pairs)

    return _resp_to_text(resp)