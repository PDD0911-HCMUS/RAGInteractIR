import json
import os
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from Database.db_session import SessionLocal
from sqlalchemy import select
from Entities.entities import RAGEmbeddingV3,IRESGVGV2, RAGEmbeddingV2
import time


MODEL_ID = "google/siglip-base-patch16-224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()


@torch.no_grad()
def embed_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)

    feats = model.get_image_features(**inputs)
    feats = F.normalize(feats, dim=-1)

    return feats.squeeze(0).cpu().tolist()


@torch.no_grad()
def embed_texts(texts):
    inputs = processor(
        text=texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)

    feats = model.get_text_features(**inputs)
    feats = F.normalize(feats, dim=-1)

    return feats.cpu()

@torch.no_grad()
def build_region_representation(
    regions: List[str],
    separator: str = " [SEP] "
):
    """
    Build one pooled text representation for all regions of one image.

    Returns:
        joined_text: str
        pooled_text_emb: List[float]
    """
    # Remove invalid / empty regions
    clean_regions = [
        str(r).strip()
        for r in regions
        if r is not None and str(r).strip() != ""
    ]

    if len(clean_regions) == 0:
        return "", None

    # Embed each region independently
    region_embs = embed_texts(clean_regions)   # [N, D]

    # Mean pooling across all region embeddings
    pooled_emb = region_embs.mean(dim=0)             # [D]

    # Re-normalize after mean pooling
    pooled_emb = F.normalize(pooled_emb, dim=0)

    # Save joined text for inspection/debugging
    joined_text = separator.join(clean_regions)

    return joined_text, pooled_emb.tolist()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_embeddings(json_path, image_root, batch_size=1000):

    data = load_json(json_path)
    
    separator = " [SEP] "
    
    # data = data[:30000]

    inserted_rows = 0
    buffer = []
    
    print(f"[LOG] DEVICE: {DEVICE}")
    print(f"[LOG] Data Length: {len(data)}")
    print(f"[LOG] Batch Size: {batch_size}")

    with SessionLocal() as session:

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        ) as progress:

            task = progress.add_task(
                "[green]Processing images...",
                total=len(data)
            )

            for item in data:

                image_id = item["image_id"]
                regions = item["regions"]

                image_path = os.path.join(image_root, f"{image_id}.jpg")

                if not os.path.exists(image_path):
                    progress.advance(task)
                    continue

                try:

                    image_emb = embed_image(image_path)
                    # Pooled region embedding
                    joined_text, pooled_text_emb = build_region_representation(
                        regions=regions,
                        separator=separator
                    )
                    # text_embs = embed_texts(regions)
                    
                    entity = RAGEmbeddingV3(
                        image_id=f"{image_id}.jpg",
                        text=joined_text,
                        image_em=image_emb,
                        text_em=pooled_text_emb
                    )
                    
                    buffer.append(entity)

                    # for region_text, text_emb in zip(regions, text_embs):

                    #     entity = RAGEmbeddingV2(
                    #         image_id=f"{image_id}.jpg",
                    #         text=region_text,
                    #         image_em=image_emb,
                    #         text_em=text_emb
                    #     )

                    #     buffer.append(entity)

                    if len(buffer) >= batch_size: # một lần add vào DB một lượng = batch_size
                        session.add_all(buffer)
                        session.commit()

                        inserted_rows += len(buffer)
                        buffer.clear()

                        progress.console.print(
                            f"[cyan]Inserted rows:[/cyan] {inserted_rows}"
                        )

                except Exception as e:
                    progress.console.print(
                        f"[red]Error image {image_id}: {e}"
                    )

                progress.advance(task)

        # insert remaining rows
        if buffer:
            session.add_all(buffer)
            session.commit()
            inserted_rows += len(buffer)

    print("\n================================")
    print("Done inserting embeddings")
    print("Total rows inserted:", inserted_rows)
    print("================================")


def get_gallery():
    print("================ Creating Gallery ================")
    with SessionLocal() as session:    
        start_te = time.time()
        db_em = session.scalars(select(RAGEmbeddingV3.image_em)).all()
        print(f"Query db_em Done {time.time() - start_te} !!!")
        print(type(db_em))
        print(len(db_em))
        print(len(db_em[0]))
        
        start_id = time.time()
        
        id_im = session.scalars(select(RAGEmbeddingV3.image_id)).all()
        print(f"Query image_id Done {time.time() - start_id} !!!")
        print(type(id_im))
        print(len(id_im))
        
        start_t = time.time()
        
        t = session.scalars(select(RAGEmbeddingV3.text)).all()
        print(f"Query text Done {time.time() - start_t} !!!")
        print(type(t))
        print(len(t))
        
        return db_em, id_im, t
    
# if __name__ == "__main__":

#     save_embeddings(
#         json_path=r"datasets\VG\top_regions.json",
#         image_root=r"datasets\VG\VG_100K",
#         batch_size=1000
#     )
    
    # text = "there are two men conversing in the photo"
    # image = os.path.join("datasets\VG\VG_100K", "1.jpg")
    # t = embed_texts(text)
    # i = embed_image(image)
    # print(len(t[0]))
    # print(len(i))
            