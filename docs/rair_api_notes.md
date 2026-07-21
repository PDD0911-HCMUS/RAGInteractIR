# RAIR API Notes

This document records the current RAIR-VF API contract used by the frontend.
Base URL in local development:

```text
http://localhost:8000/api/v1
```

## Runtime Defaults

The API service is configured by environment variables. Current recommended setup:

```bash
RAIR_LLM_PROVIDER=local
RAIR_REASONING_MODEL=google/gemma-3-12b-it
LOCAL_LLM_DEVICE=cuda
LOCAL_LLM_DTYPE=bfloat16

RAIR_EMBEDDING_BACKEND=siglip
RAIR_EMBEDDING_MODEL=google/siglip-base-patch16-224
RAIR_EMBEDDING_MODE=train
RAIR_RETRIEVAL_INDEX=fusion
RAIR_FUSION_ALPHA=0.9
RAIR_FUSION_POOL_SIZE=200

RAIR_EVIDENCE_TOP_K=10
RAIR_FACT_TOP_M=4
PRELOAD_RAIR_API=1
PRELOAD_LEGACY_VLM=0
```

The frontend can override the retrieval backend per session when creating a new session.

## 1. Create Empty RAIR Session

```http
POST /rair/sessions
```

Creates a new RAIR session but does not run retrieval yet. Use this when the user starts a new retrieval conversation.

### Request Body

Minimal body:

```json
{}
```

Full body:

```json
{
  "embedding_backend": "siglip",
  "embedding_model": "google/siglip-base-patch16-224",
  "retrieval_index": "fusion",
  "fusion_alpha": 0.9,
  "fusion_pool_size": 200
}
```

### Fields

| Field | Type | Required | Notes |
|---|---|---:|---|
| `embedding_backend` | string | no | `siglip` or `clip`. Default comes from `RAIR_EMBEDDING_BACKEND`. |
| `embedding_model` | string | no | Optional model override. For CLIP use `openai/clip-vit-base-patch32`; for SigLIP use `google/siglip-base-patch16-224`. |
| `retrieval_index` | string | no | `image`, `caption`, or `fusion`. |
| `fusion_alpha` | number | no | Image score weight when `retrieval_index=fusion`. Must be in `[0.0, 1.0]`. |
| `fusion_pool_size` | integer | no | Candidate pool size for fusion reranking. |

### Response

```json
{
  "session_id": "uuid",
  "session": {
    "session_id": "uuid",
    "initial_query": "",
    "current_query": "",
    "turn": 0,
    "embedding_backend": "siglip",
    "embedding_model": "google/siglip-base-patch16-224",
    "retrieval_index": "fusion",
    "fusion_alpha": 0.9,
    "fusion_pool_size": 200,
    "feedback_pairs": [],
    "pending_suggestions": []
  }
}
```

## 2. Send Query Or Feedback Turn

```http
POST /rair/sessions/{session_id}/turns
```

This endpoint is used for both the first query and later feedback turns.

- If `turn == 0`, `message` is treated as the initial user query.
- If `turn > 0`, `message` is treated as user feedback to the previous RAIR suggestions.

### Request Body

```json
{
  "message": "a man riding a snowboard"
}
```

### Response

```json
{
  "session_id": "uuid",
  "session": {
    "session_id": "uuid",
    "initial_query": "a man riding a snowboard",
    "current_query": "man snowboarding on snowy slope",
    "turn": 1,
    "embedding_backend": "siglip",
    "embedding_model": "google/siglip-base-patch16-224",
    "retrieval_index": "fusion",
    "fusion_alpha": 0.9,
    "fusion_pool_size": 200,
    "feedback_pairs": [],
    "pending_suggestions": [
      {
        "sug": "red snowboard",
        "type": "add_detail",
        "explain": "..."
      }
    ]
  },
  "turn": {
    "turn": 1,
    "user_message": "a man riding a snowboard",
    "context_state": {},
    "rewritten_query": "man snowboarding on snowy slope",
    "retrieval": {
      "backend": "siglip",
      "model": "google/siglip-base-patch16-224",
      "index": "fusion",
      "fusion_alpha": 0.9,
      "fusion_pool_size": 200,
      "search_depth": 50,
      "results": [
        {
          "rank": 1,
          "image_id": "train2014/COCO_train2014_000000xxxxxx.jpg",
          "caption": "...",
          "score": 0.123,
          "media_url": "/media/train2014/COCO_train2014_000000xxxxxx.jpg"
        }
      ]
    },
    "candidate_evidence": [],
    "fact_selection": {
      "method": "qvfs",
      "evidence_top_k": 10,
      "top_m": 4,
      "alpha": 0.5,
      "beta": 0.3,
      "gamma": 0.2,
      "delta": 0.5
    },
    "diagnosis": {},
    "suggestions": [
      {
        "sug": "red snowboard",
        "type": "add_detail",
        "explain": "..."
      }
    ],
    "meta": {
      "llm_provider": "local",
      "reasoning_model": "google/gemma-3-12b-it",
      "rewrite_latency_ms": 1234,
      "elapsed_ms": 5678
    }
  }
}
```

## 3. Get Session State

```http
GET /rair/sessions/{session_id}
```

Returns the session state and full turn history. No request body.

### Response

```json
{
  "session_id": "uuid",
  "initial_query": "a man riding a snowboard",
  "current_query": "man snowboarding on snowy slope",
  "turn": 1,
  "embedding_backend": "siglip",
  "embedding_model": "google/siglip-base-patch16-224",
  "retrieval_index": "fusion",
  "fusion_alpha": 0.9,
  "fusion_pool_size": 200,
  "feedback_pairs": [],
  "pending_suggestions": [],
  "history": []
}
```

## Frontend Flow

```text
1. User opens a new retrieval session.
   POST /api/v1/rair/sessions
   Save response.session_id.

2. User submits the first query.
   POST /api/v1/rair/sessions/{session_id}/turns
   Body: { "message": initialQuery }
   Render turn.retrieval.results and turn.suggestions.

3. User accepts, edits, or adds feedback.
   POST /api/v1/rair/sessions/{session_id}/turns
   Body: { "message": feedbackText }
   Render the new turn.

4. Optional debug/state refresh.
   GET /api/v1/rair/sessions/{session_id}
```

## Important Notes

- Do not call `GET /api/v1/rair/sessions`; this route does not exist.
- `POST /api/v1/rair/sessions` only creates an empty session.
- The first retrieval happens at `POST /api/v1/rair/sessions/{session_id}/turns`.
- Existing legacy endpoints under `/api/v1/vlm/...` are separate from the RAIR API.
- If frontend base URL ends with `/`, avoid adding another leading slash to the path to prevent URLs such as `http://localhost:8000//api/v1/...`.

## Curl Examples

Create session:

```bash
curl -X POST http://localhost:8000/api/v1/rair/sessions \
  -H "Content-Type: application/json" \
  -d '{"embedding_backend":"siglip","retrieval_index":"fusion","fusion_alpha":0.9}'
```

Send first query:

```bash
curl -X POST http://localhost:8000/api/v1/rair/sessions/{session_id}/turns \
  -H "Content-Type: application/json" \
  -d '{"message":"a man riding a snowboard"}'
```

Send feedback:

```bash
curl -X POST http://localhost:8000/api/v1/rair/sessions/{session_id}/turns \
  -H "Content-Type: application/json" \
  -d '{"message":"the snowboard is red and the person wears a black jacket"}'
```
