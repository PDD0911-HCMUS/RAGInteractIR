import threading
import traceback
from typing import Optional, List, Tuple, Literal

from lmdeploy import PytorchEngineConfig, pipeline

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
                try:
                    print(">>> before pipeline()")
                    _pipe = pipeline(
                        MODEL_NAME,
                        backend_config=PytorchEngineConfig(
                            # session_len=4096,
                            max_batch_size=1,
                            cache_max_entry_count=0.2,
                        ),
                    )
                    print(">>> after pipeline()")
                except Exception as e:
                    print(">>> pipeline() failed:", repr(e))
                    traceback.print_exc()
                    raise
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
    # print(f"Norm Hist Func: {history}")
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

def generate_answer(
    user_prompt: str,
    history: Optional[RoleHistory] = None
):
    pipe = get_pipe()
    hist_pairs = normalize_history_for_lmdeploy(history)
    # print(f"Normalize hist_pairs: {hist_pairs}")
    resp = pipe(user_prompt, history=hist_pairs)
    return _resp_to_text(resp)