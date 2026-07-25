from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from sqlalchemy import text

from Database.create_flickr30k_tables import CREATE_FLICKR30K_TARGET_SQL, INDEX_SQL
from Database.db_session import engine

try:
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
except Exception:  # pragma: no cover
    Progress = None


INSERT_TARGET_SQL = """
INSERT INTO "Flickr30KTargetAnnotations"
    ("image_id", "image_path", "split", "base_caption", "captions",
     "visual_facts", "positive_facts", "negative_facts", "uncertain_facts",
     "entity_phrases", "grounded_phrases", "boxes", "enriched_caption", "source")
VALUES
    (:image_id, :image_path, :split, :base_caption, :captions,
     :visual_facts, :positive_facts, :negative_facts, :uncertain_facts,
     :entity_phrases, :grounded_phrases, :boxes, :enriched_caption, :source)
"""

EXISTING_TARGET_SQL = """
SELECT "image_id"
FROM "Flickr30KTargetAnnotations"
WHERE "split" = :split
"""

CAPTION_ENTITY_RE = re.compile(r"\[/EN#(?P<chain_id>[^/\s\]]+)(?P<types>(?:/[^ \]]+)*) (?P<phrase>[^\]]+)\]")

STOP_PHRASES = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "image",
    "photo",
    "picture",
}


def normalize_space(text_value: Any) -> str:
    return " ".join(str(text_value or "").strip().split())


def dedupe_keep_order(items: Iterable[Any]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        clean = normalize_space(item)
        key = clean.lower()
        if not clean or key in STOP_PHRASES or key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def load_split_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        ids.append(Path(value).stem)
    return ids


def discover_image_ids(root: Path, split: str) -> List[str]:
    split_ids = load_split_ids(root / f"{split}.txt")
    if split_ids:
        return split_ids

    sentence_dir = root / "Sentences"
    return sorted(path.stem for path in sentence_dir.glob("*.txt"))


def clean_caption_line(line: str) -> str:
    return normalize_space(CAPTION_ENTITY_RE.sub(lambda match: match.group("phrase"), line))


def parse_sentences(path: Path) -> Dict[str, Any]:
    captions: List[str] = []
    phrases: List[str] = []
    phrase_records: List[Dict[str, Any]] = []

    if not path.exists():
        return {"captions": [], "entity_phrases": [], "phrase_records": []}

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = normalize_space(raw_line)
        if not line:
            continue
        captions.append(clean_caption_line(line))
        for match in CAPTION_ENTITY_RE.finditer(line):
            phrase = normalize_space(match.group("phrase"))
            chain_id = match.group("chain_id")
            types = [value for value in match.group("types").split("/") if value]
            phrases.append(phrase)
            phrase_records.append(
                {
                    "chain_id": chain_id,
                    "types": types,
                    "phrase": phrase,
                    "is_visual": "notvisual" not in types and chain_id != "0",
                }
            )

    return {
        "captions": captions,
        "entity_phrases": dedupe_keep_order(phrases),
        "phrase_records": phrase_records,
    }


def parse_annotation_xml(path: Path) -> Dict[str, Any]:
    boxes: List[Dict[str, Any]] = []
    grounded_chain_ids: Set[str] = set()
    scene_chain_ids: Set[str] = set()
    nobox_chain_ids: Set[str] = set()

    if not path.exists():
        return {
            "boxes": [],
            "grounded_chain_ids": [],
            "scene_chain_ids": [],
            "nobox_chain_ids": [],
        }

    root = ET.parse(path).getroot()
    for obj in root.findall("object"):
        names = [normalize_space(node.text) for node in obj.findall("name")]
        names = [name for name in names if name]
        if not names:
            continue

        scene = normalize_space(obj.findtext("scene")) == "1"
        nobox = normalize_space(obj.findtext("nobndbox")) == "1"
        bndbox = obj.find("bndbox")

        if scene:
            scene_chain_ids.update(names)
        if nobox:
            nobox_chain_ids.update(names)

        if bndbox is None:
            continue

        box = {
            "chain_ids": names,
            "xmin": int(float(normalize_space(bndbox.findtext("xmin")) or 0)),
            "ymin": int(float(normalize_space(bndbox.findtext("ymin")) or 0)),
            "xmax": int(float(normalize_space(bndbox.findtext("xmax")) or 0)),
            "ymax": int(float(normalize_space(bndbox.findtext("ymax")) or 0)),
        }
        boxes.append(box)
        grounded_chain_ids.update(names)

    return {
        "boxes": boxes,
        "grounded_chain_ids": sorted(grounded_chain_ids),
        "scene_chain_ids": sorted(scene_chain_ids),
        "nobox_chain_ids": sorted(nobox_chain_ids),
    }


def build_visual_facts(sentence_data: Dict[str, Any], annotation_data: Dict[str, Any]) -> Dict[str, List[str]]:
    grounded_ids = set(annotation_data["grounded_chain_ids"])
    scene_ids = set(annotation_data["scene_chain_ids"])
    nobox_ids = set(annotation_data["nobox_chain_ids"])

    entity_phrases = []
    grounded_phrases = []
    uncertain_phrases = []

    for record in sentence_data["phrase_records"]:
        phrase = record["phrase"]
        chain_id = record["chain_id"]
        if not record["is_visual"]:
            uncertain_phrases.append(phrase)
            continue
        entity_phrases.append(phrase)
        if chain_id in grounded_ids:
            grounded_phrases.append(phrase)
        elif chain_id in scene_ids or chain_id in nobox_ids:
            entity_phrases.append(phrase)
        else:
            uncertain_phrases.append(phrase)

    captions = sentence_data["captions"]
    visual_facts = dedupe_keep_order([*grounded_phrases, *entity_phrases])
    if captions:
        visual_facts = dedupe_keep_order([captions[0], *visual_facts])

    return {
        "entity_phrases": dedupe_keep_order(entity_phrases),
        "grounded_phrases": dedupe_keep_order(grounded_phrases),
        "visual_facts": visual_facts,
        "positive_facts": visual_facts,
        "negative_facts": [],
        "uncertain_facts": dedupe_keep_order(uncertain_phrases),
    }


class Flickr30KEntitiesBuilder:
    def __init__(
        self,
        root: str,
        image_folder: str,
        split: str,
        image_ext: str = ".jpg",
        image_subdir: str = "",
    ) -> None:
        self.root = Path(root)
        self.image_folder = Path(image_folder)
        self.split = split
        self.image_ext = image_ext
        self.image_subdir = image_subdir.strip("/\\")
        self.sentence_dir = self.root / "Sentences"
        self.annotation_dir = self.root / "Annotations"

    def ensure_table(self) -> None:
        with engine.begin() as conn:
            conn.execute(text(CREATE_FLICKR30K_TARGET_SQL))
            for sql in INDEX_SQL:
                if "Flickr30KTargetAnnotations" in sql:
                    conn.execute(text(sql))

    def load_existing_ids(self) -> Set[str]:
        with engine.begin() as conn:
            rows = conn.execute(text(EXISTING_TARGET_SQL), {"split": self.split}).fetchall()
        return {str(row[0]) for row in rows}

    def image_path_for(self, image_id: str) -> str:
        filename = f"{image_id}{self.image_ext}"
        return str(Path(self.image_subdir) / filename).replace("\\", "/") if self.image_subdir else filename

    def build_records(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        skip_existing: bool = False,
    ) -> int:
        self.ensure_table()
        ids = discover_image_ids(self.root, self.split)[offset:]
        if limit is not None:
            ids = ids[:limit]
        existing = self.load_existing_ids() if skip_existing else set()

        inserted = 0
        iterator = ids
        progress = None
        task = None
        if Progress is not None:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeRemainingColumn(),
            )
            progress.__enter__()
            task = progress.add_task("[green]Parsing Flickr30K Entities...", total=len(ids))

        try:
            for image_id in iterator:
                if image_id in existing:
                    if progress is not None:
                        progress.advance(task)
                    continue

                sentence_data = parse_sentences(self.sentence_dir / f"{image_id}.txt")
                annotation_data = parse_annotation_xml(self.annotation_dir / f"{image_id}.xml")
                facts = build_visual_facts(sentence_data, annotation_data)
                captions = sentence_data["captions"]
                image_path = self.image_path_for(image_id)
                enriched_caption = (
                    f"{captions[0]}. Selected visual facts: " + "; ".join(facts["visual_facts"])
                    if captions
                    else "; ".join(facts["visual_facts"])
                )

                payload = {
                    "image_id": image_id,
                    "image_path": image_path,
                    "split": self.split,
                    "base_caption": captions[0] if captions else "",
                    "captions": captions,
                    "visual_facts": facts["visual_facts"],
                    "positive_facts": facts["positive_facts"],
                    "negative_facts": facts["negative_facts"],
                    "uncertain_facts": facts["uncertain_facts"],
                    "entity_phrases": facts["entity_phrases"],
                    "grounded_phrases": facts["grounded_phrases"],
                    "boxes": annotation_data["boxes"],
                    "enriched_caption": enriched_caption,
                    "source": "flickr30k_entities",
                }
                with engine.begin() as conn:
                    conn.execute(text(INSERT_TARGET_SQL), payload)
                inserted += 1
                existing.add(image_id)

                if progress is not None:
                    progress.advance(task)
        finally:
            if progress is not None:
                progress.__exit__(None, None, None)

        print(f"Inserted Flickr30KTargetAnnotations rows: {inserted}")
        return inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse Flickr30K Entities into DB visual facts.")
    parser.add_argument("--root", required=True, help="Folder containing Sentences/, Annotations/, and split txt files.")
    parser.add_argument("--image-folder", required=True, help="Image folder root, used for path validation conventions.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--image-ext", default=".jpg")
    parser.add_argument("--image-subdir", default="", help="Optional relative subdirectory stored before image filename.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = Flickr30KEntitiesBuilder(
        root=args.root,
        image_folder=args.image_folder,
        split=args.split,
        image_ext=args.image_ext,
        image_subdir=args.image_subdir,
    )
    builder.build_records(
        limit=args.limit,
        offset=args.offset,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
