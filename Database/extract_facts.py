import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DEFAULT_VISDIAL_DIR = Path(r"F:\RAGInteractIR\datasets\VisDial")
DEFAULT_OUTPUT_DIR = Path(r"F:\RAGInteractIR\datasets\rair")
DEFAULT_MAP_PATH = DEFAULT_VISDIAL_DIR / "coco2014_id_to_relpath.json"
SOURCE_NAME = "rule_based_v1"

YES_PREFIXES = ("yes", "yeah", "yep")
NO_PREFIXES = ("no", "nope", "not", "i don't", "there isn't", "there aren't")
UNCERTAIN_MARKERS = (
    "maybe",
    "probably",
    "possibly",
    "looks like",
    "appears",
    "seems",
    "i think",
    "i can't tell",
    "can't tell",
    "not sure",
    "hard to tell",
)


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ?.!;:,")


def clean_answer_value(answer: str) -> str:
    a = normalize_text(answer)
    a = re.sub(r"^(it is|it's|they are|there is|there are)\s+", "", a)
    return a


def strip_question_prefix(question: str) -> str:
    q = normalize_text(question)
    q = re.sub(r"^(is|are|was|were|do|does|did|can|could|would|has|have|had)\s+", "", q)
    q = re.sub(r"^(there|this|that|it|the image|the photo|the picture)\s+", "", q)
    q = q.replace(" any ", " ")
    return q.strip()


def clean_entity(text: str) -> str:
    value = normalize_text(text)
    value = re.sub(r"\bany\s+", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def verbalize_clause(text: str) -> str:
    value = clean_entity(text)
    value = re.sub(r"^(he|she|it|they|the subject|the subjects)\s+in\s+", r"\1 is in ", value)
    value = re.sub(r"^(he|she|it|they|the subject|the subjects)\s+wearing\s+", r"\1 is wearing ", value)
    value = re.sub(r"^(the sun)\s+out$", r"\1 is out", value)
    value = value.replace("they is ", "they are ")
    value = value.replace("the subjects is ", "the subjects are ")
    return value


def is_yes_answer(answer: str) -> bool:
    a = normalize_text(answer)
    return a.startswith(YES_PREFIXES)


def is_no_answer(answer: str) -> bool:
    a = normalize_text(answer)
    return a.startswith(NO_PREFIXES)


def is_uncertain_answer(answer: str) -> bool:
    a = normalize_text(answer)
    return any(marker in a for marker in UNCERTAIN_MARKERS)


def subject_from_or_question(question: str) -> str:
    q = normalize_text(question)
    if "child or adult" in q:
        return "the person"
    if q in {"male or female"}:
        return "the person"

    subject_match = re.match(
        r"^(?:is|are|was|were)\s+(.+?)\s+(?:inside|outside|carpeted|wooden|male|female|child|adult)\s+or\s+.+$",
        q,
    )
    if subject_match:
        return subject_match.group(1).replace("this", "the subject").replace("they", "the subjects").strip()

    before_or = q.split(" or ", 1)[0]
    before_or = re.sub(r"^(is|are|was|were)\s+", "", before_or)
    before_or = before_or.replace("this", "the subject").replace("they", "the subjects")
    return before_or.strip()


def build_positive_fact(question: str, answer: str) -> str:
    q = strip_question_prefix(question)
    original_q = normalize_text(question)
    a = clean_answer_value(answer)

    color_match = re.match(r"^what(?:'s| is)? (?:the )?color (?:of|is) (.+)$", original_q)
    if color_match:
        return f"{color_match.group(1).strip()} is {a}"

    breed_match = re.match(r"^what breed is (.+)$", original_q)
    if breed_match:
        return f"{breed_match.group(1).strip()} is a {a}"

    made_of_match = re.match(r"^what(?:'s| is) (.+) made out of$", original_q)
    if made_of_match:
        return f"{made_of_match.group(1).strip()} is made of {a}"

    if " or " in original_q and not is_yes_answer(a):
        subject = subject_from_or_question(original_q)
        if subject:
            verb = "are" if subject in {"they", "the subjects"} else "is"
            return f"{subject} {verb} {a}"

    if original_q.startswith("is there ") and is_yes_answer(a):
        entity = clean_entity(original_q.removeprefix("is there "))
        extra = normalize_text(answer).split(",", 1)[1].strip() if "," in normalize_text(answer) else ""
        return f"there is {entity}: {extra}" if extra else f"there is {entity}"

    if original_q.startswith("are there ") and is_yes_answer(a):
        entity = clean_entity(original_q.removeprefix("are there "))
        extra = normalize_text(answer).split(",", 1)[1].strip() if "," in normalize_text(answer) else ""
        return f"there are {entity}: {extra}" if extra else f"there are {entity}"

    if is_yes_answer(a):
        return verbalize_clause(q)

    if q.startswith(("what ", "where ", "who ", "how many ", "which ")):
        return f"{q}: {a}"

    return f"{q}; answer: {a}"


def build_negative_fact(question: str, answer: str) -> str:
    q = strip_question_prefix(question)
    original_q = normalize_text(question)
    a = normalize_text(answer)

    if original_q.startswith("is there "):
        return "there is no " + clean_entity(original_q.removeprefix("is there "))

    if original_q.startswith("are there "):
        return "there are no " + clean_entity(original_q.removeprefix("are there "))

    if is_no_answer(a):
        clause = verbalize_clause(q)
        if clause.startswith(("he is ", "she is ", "it is ", "they are ")):
            return clause.replace(" is ", " is not ", 1).replace(" are ", " are not ", 1)
        if clause.startswith("the sun is "):
            return clause.replace(" is ", " is not ", 1)
        return f"not {clause}"

    return f"{q}; negative answer: {a}"


def classify_fact(question: str, answer: str) -> Dict[str, str]:
    q = normalize_text(question)
    a = normalize_text(answer)

    if is_uncertain_answer(a):
        return {
            "polarity": "uncertain",
            "fact": build_positive_fact(q, a),
        }

    if is_no_answer(a):
        return {
            "polarity": "negative",
            "fact": build_negative_fact(q, a),
        }

    return {
        "polarity": "positive",
        "fact": build_positive_fact(q, a),
    }


def resolve_dialog(
    dialog_index: int,
    dialog_item: Dict[str, Any],
    questions: List[str],
    answers: List[str],
    image_map: Dict[str, str],
    split: str,
) -> Dict[str, Any]:
    image_id = dialog_item.get("image_id")
    image_id_key = str(image_id)
    base_caption = normalize_text(dialog_item.get("caption"))

    dialogue = []
    visual_facts = []
    positive_facts = []
    negative_facts = []
    uncertain_facts = []

    if base_caption:
        visual_facts.append(base_caption)
        positive_facts.append(base_caption)

    for round_id, turn in enumerate(dialog_item.get("dialog", []), start=1):
        question = questions[turn["question"]]
        answer = answers[turn["answer"]]
        fact = classify_fact(question, answer)

        dialogue.append(
            {
                "round": round_id,
                "question": question,
                "answer": answer,
                "fact": fact["fact"],
                "polarity": fact["polarity"],
            }
        )

        if fact["fact"]:
            visual_facts.append(fact["fact"])
            if fact["polarity"] == "negative":
                negative_facts.append(fact["fact"])
            elif fact["polarity"] == "uncertain":
                uncertain_facts.append(fact["fact"])
            else:
                positive_facts.append(fact["fact"])

    enriched_caption = build_enriched_caption(base_caption, positive_facts, negative_facts, uncertain_facts)

    return {
        "dialog_index": dialog_index,
        "image_id": image_id,
        "image_path": resolve_image_path(image_id_key, image_map, split),
        "base_caption": base_caption,
        "dialogue": dialogue,
        "visual_facts": dedupe_keep_order(visual_facts),
        "positive_facts": dedupe_keep_order(positive_facts),
        "negative_facts": dedupe_keep_order(negative_facts),
        "uncertain_facts": dedupe_keep_order(uncertain_facts),
        "enriched_caption": enriched_caption,
    }


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        clean = normalize_text(item)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def resolve_image_path(image_id: str, image_map: Dict[str, str], split: str) -> Optional[str]:
    mapped = image_map.get(str(image_id))
    if mapped:
        return mapped

    if split == "val":
        return f"VisualDialog_val2018/VisualDialog_val2018_{int(image_id):012d}.jpg"

    return None


def build_enriched_caption(
    base_caption: str,
    positive_facts: List[str],
    negative_facts: List[str],
    uncertain_facts: List[str],
) -> str:
    pieces = []
    positives = [fact for fact in dedupe_keep_order(positive_facts) if fact != base_caption]
    negatives = dedupe_keep_order(negative_facts)
    uncertain = dedupe_keep_order(uncertain_facts)

    if base_caption:
        pieces.append(base_caption)
    if positives:
        pieces.append("Additional visual facts: " + "; ".join(positives))
    if negatives:
        pieces.append("Negative visual facts: " + "; ".join(negatives))
    if uncertain:
        pieces.append("Uncertain visual facts: " + "; ".join(uncertain))

    return ". ".join(pieces).strip()


def load_visdial(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "data" not in data:
        raise ValueError(f"Invalid VisDial file, missing data field: {path}")

    return data


def load_image_map(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_records(
    input_path: Path,
    image_map: Dict[str, str],
    split: str,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    raw = load_visdial(input_path)
    data = raw["data"]
    questions = data.get("questions", [])
    answers = data.get("answers", [])
    dialogs = data.get("dialogs", [])

    if offset:
        dialogs = dialogs[offset:]
    if limit is not None:
        dialogs = dialogs[:limit]

    records = []
    for local_index, dialog_item in enumerate(dialogs):
        dialog_index = offset + local_index
        record = resolve_dialog(
            dialog_index=dialog_index,
            dialog_item=dialog_item,
            questions=questions,
            answers=answers,
            image_map=image_map,
            split=split,
        )
        record["split"] = split
        record["source"] = SOURCE_NAME
        records.append(record)

    return records


def write_jsonl(records: List[Dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(records)


def to_db_entity(record: Dict[str, Any]):
    from Entities.entities import VisDialTargetAnnotations

    return VisDialTargetAnnotations(
        split=record.get("split"),
        dialog_index=record.get("dialog_index"),
        image_id=str(record.get("image_id")) if record.get("image_id") is not None else None,
        image_path=record.get("image_path"),
        base_caption=record.get("base_caption"),
        dialogue=record.get("dialogue"),
        visual_facts=record.get("visual_facts"),
        positive_facts=record.get("positive_facts"),
        negative_facts=record.get("negative_facts"),
        uncertain_facts=record.get("uncertain_facts"),
        enriched_caption=record.get("enriched_caption"),
        source=record.get("source", SOURCE_NAME),
        created_at=datetime.now(timezone.utc),
    )


def delete_existing_split(session, split: str) -> int:
    from Entities.entities import VisDialTargetAnnotations

    return (
        session.query(VisDialTargetAnnotations)
        .filter(VisDialTargetAnnotations.split == split)
        .delete(synchronize_session=False)
    )


def write_db(
    records: List[Dict[str, Any]],
    split: str,
    replace: bool = False,
    batch_size: int = 1000,
) -> int:
    from Database.db_session import SessionLocal

    inserted = 0
    with SessionLocal() as session:
        if replace:
            deleted = delete_existing_split(session, split)
            session.commit()
            print(f"[{split}] deleted {deleted} existing DB records")

        buffer = []
        for record in records:
            buffer.append(to_db_entity(record))
            if len(buffer) >= batch_size:
                session.add_all(buffer)
                session.commit()
                inserted += len(buffer)
                buffer.clear()
                print(f"[{split}] inserted {inserted} DB records")

        if buffer:
            session.add_all(buffer)
            session.commit()
            inserted += len(buffer)

    return inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract RAIR target visual facts from VisDial annotations."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_VISDIAL_DIR / "visdial_1.0_train.json",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=DEFAULT_VISDIAL_DIR / "visdial_1.0_val.json",
    )
    parser.add_argument("--image-map", type=Path, default=DEFAULT_MAP_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--write-db",
        action="store_true",
        help="Insert extracted records into VisDialTargetAnnotations.",
    )
    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Do not write JSONL files.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing DB rows for each processed split before inserting.",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_map = load_image_map(args.image_map)

    jobs = [
        ("train", args.train, args.output_dir / "target_annotations_train.jsonl"),
        ("val", args.val, args.output_dir / "target_annotations_val.jsonl"),
    ]

    for split, input_path, output_path in jobs:
        records = extract_records(
            input_path=input_path,
            image_map=image_map,
            split=split,
            limit=args.limit,
            offset=args.offset,
        )

        if not args.no_jsonl:
            count = write_jsonl(records, output_path)
            print(f"[{split}] wrote {count} records to {output_path}")

        if args.write_db:
            count = write_db(
                records=records,
                split=split,
                replace=args.replace,
                batch_size=args.batch_size,
            )
            print(f"[{split}] inserted {count} records into VisDialTargetAnnotations")


if __name__ == "__main__":
    main()
