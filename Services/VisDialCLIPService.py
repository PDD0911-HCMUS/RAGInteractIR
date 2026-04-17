import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, CLIPModel, AutoProcessor
import torch.nn.functional as F
import time
from Database.db_session import SessionLocal
from Entities.entities import VisDialCLIPCapDial #, VisDialCLIPDial, VisDialCLIPAnswers, VisDialCLIPQuestions
from sqlalchemy import select

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)

class VisDialCLIPService:
    
    def __init__(self, vlm, device):
        
        self.vlm = vlm
        self.device = device
        self.tokenizer, self.processor, self.model = self.create_vlm()
        pass
    
    def create_vlm(self):
        tokenizer = AutoTokenizer.from_pretrained(self.vlm)
        model = CLIPModel.from_pretrained(self.vlm).to(self.device).eval()
        processor = AutoProcessor.from_pretrained(self.vlm)
        
        return tokenizer, processor, model
    
    @torch.no_grad()
    def embed_texts(self, texts):
        inputs = self.tokenizer(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        feats = self.model.get_text_features(**inputs)
        feats = F.normalize(feats, dim=-1)

        return feats.cpu()
    
    def build_gallery(self):
        console = Console()
        start_total = time.perf_counter()

        console.rule("[bold cyan]Creating VisDial Gallery + FAISS[/bold cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Initializing...", total=5)

            # Step 1: Load rows from DB
            progress.update(task, description="Loading rows from database...", completed=0)
            t0 = time.perf_counter()
            with SessionLocal() as session:
                rows = session.execute(
                    select(
                        VisDialCLIPCapDial.image_path,
                        VisDialCLIPCapDial.caption,
                        VisDialCLIPCapDial.img_em,
                        VisDialCLIPCapDial.cap_em,
                    )
                ).all()
            t1 = time.perf_counter()
            progress.advance(task)

            if not rows:
                console.print("[bold red]No data found in VisDialCLIPCapDial.[/bold red]")
                raise ValueError("No data found in VisDialCLIPCapDial.")

            # Step 2: Extract metadata
            progress.update(task, description="Extracting image paths and captions...", completed=1)
            t2 = time.perf_counter()
            image_ids = [row.image_path for row in rows]
            captions = [row.caption for row in rows]
            t3 = time.perf_counter()
            progress.advance(task)

            # Step 3: Convert embeddings to numpy
            progress.update(task, description="Converting embeddings to numpy arrays...", completed=2)
            t4 = time.perf_counter()
            img_embeddings = np.array([row.img_em for row in rows], dtype=np.float32)
            cap_embeddings = np.array([row.cap_em for row in rows], dtype=np.float32)
            cap_embeddings = np.squeeze(cap_embeddings, axis=1)
            t5 = time.perf_counter()
            progress.advance(task)

            # Optional: normalize if needed for cosine-like IP search
            progress.update(task, description="Preparing FAISS indices...", completed=3)
            t6 = time.perf_counter()

            # Uncomment if your embeddings are NOT already normalized
            # faiss.normalize_L2(img_embeddings)
            # faiss.normalize_L2(cap_embeddings)

            dim_img = img_embeddings.shape[1]
            dim_cap = cap_embeddings.shape[1]

            img_index = faiss.IndexFlatIP(dim_img)
            cap_index = faiss.IndexFlatIP(dim_cap)

            img_index.add(img_embeddings)
            cap_index.add(cap_embeddings)
            t7 = time.perf_counter()
            progress.advance(task)

            # Step 5: Finalize gallery
            progress.update(task, description="Finalizing gallery object...", completed=4)
            t8 = time.perf_counter()
            gallery = {
                "image_id": image_ids,
                "caption": captions,
                "img_em": img_embeddings,
                "cap_em": cap_embeddings,
                "img_index": img_index,
                "cap_index": cap_index,
            }
            t9 = time.perf_counter()
            progress.advance(task)

        # Summary table
        summary = Table(title="Gallery Build Summary", show_lines=False)
        summary.add_column("Field", style="cyan", no_wrap=True)
        summary.add_column("Value", style="green")

        summary.add_row("Loaded rows", f"{len(rows):,}")
        summary.add_row("Image embedding shape", str(img_embeddings.shape))
        summary.add_row("Caption embedding shape", str(cap_embeddings.shape))
        summary.add_row("FAISS image index total", f"{img_index.ntotal:,}")
        summary.add_row("FAISS caption index total", f"{cap_index.ntotal:,}")
        summary.add_row("DB load time", f"{t1 - t0:.2f}s")
        summary.add_row("Metadata extract time", f"{t3 - t2:.2f}s")
        summary.add_row("Numpy conversion time", f"{t5 - t4:.2f}s")
        summary.add_row("FAISS build time", f"{t7 - t6:.2f}s")
        summary.add_row("Finalize time", f"{t9 - t8:.2f}s")
        summary.add_row("Total time", f"{time.perf_counter() - start_total:.2f}s")

        console.print(summary)
        console.rule("[bold green]Build done[/bold green]")

        return gallery
    
    def faiss_search(self, query_text, gallery, top_k=20):
        q = self.embed_texts(query_text)
        q = np.array(q, dtype=np.float32).reshape(1, -1)
        # faiss.normalize_L2(q)

        scores, indices = gallery["img_index"].search(q, top_k) # SIM(q, image)

        # results = []
        image_ids, captions, s = [], [], []
        for score, idx in zip(scores[0], indices[0]):
            # results.append({
            #     "image_id": gallery["image_id"][idx],
            #     "caption": gallery["caption"][idx],
            #     "score": float(score)
            # })
            image_ids.append(gallery["image_id"][idx])
            captions.append(gallery["caption"][idx])
            s.append(float(score))
        return image_ids, captions, s
        
    