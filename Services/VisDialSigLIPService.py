import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
import torch.nn.functional as F

from Database.db_session import SessionLocal
from Entities.entities import VisDialSigLipCapDial, VisDialSigLipDial, VisDialSigLipAnswers, VisDialSigLipQuestions
from sqlalchemy import select

class VisDialSiGLIPService:
    
    def __init__(self, vlm, device):
        
        self.vlm = vlm
        self.device = device
        self.processor, self.model = self.create_vlm()
        pass
    
    def create_vlm(self):
        processor = AutoProcessor.from_pretrained(self.vlm)
        model = AutoModel.from_pretrained(self.vlm).to(self.device).eval()
        
        return processor, model
    
    @torch.no_grad()
    def embed_texts(self, texts):
        inputs = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        feats = self.model.get_text_features(**inputs)
        feats = F.normalize(feats, dim=-1)

        return feats.cpu()
    
    def build_gallery(self):
        print("================ Creating VisDial Gallery + FAISS ================")

        with SessionLocal() as session:
            rows = session.execute(
                select(
                    VisDialSigLipCapDial.image_path,
                    VisDialSigLipCapDial.caption,
                    VisDialSigLipCapDial.img_em,
                    VisDialSigLipCapDial.cap_em
                )
            ).all()

        if not rows:
            raise ValueError("No data found in VisDialSigLipCapDial.")

        image_ids = [row.image_path for row in rows]
        captions = [row.caption for row in rows]

        # DB -> numpy
        # giả sử mỗi row.img_em / row.cap_em đã là list hoặc array-like
        img_embeddings = np.array([row.img_em for row in rows], dtype=np.float32)
        cap_embeddings = np.array([row.cap_em for row in rows], dtype=np.float32)
        cap_embeddings = np.squeeze(cap_embeddings, axis=1)
        print(f"Loaded rows           : {len(rows)}")
        print(f"Image embedding shape : {img_embeddings.shape}")
        print(f"Caption embedding shape: {cap_embeddings.shape}")

        # normalize để dùng inner product như cosine similarity
        # faiss.normalize_L2(img_embeddings)
        # faiss.normalize_L2(cap_embeddings)

        dim_img = img_embeddings.shape[1]
        dim_cap = cap_embeddings.shape[1]

        # IndexFlatIP: exact search, dùng với normalized vector ~ cosine similarity
        img_index = faiss.IndexFlatIP(dim_img)
        cap_index = faiss.IndexFlatIP(dim_cap)

        img_index.add(img_embeddings)
        cap_index.add(cap_embeddings)

        print(f"FAISS image index total vectors   : {img_index.ntotal}")
        print(f"FAISS caption index total vectors : {cap_index.ntotal}")
        print("================ Build done ================")

        gallery = {
            "image_id": image_ids,
            "caption": captions,
            "img_em": img_embeddings,
            "cap_em": cap_embeddings,
            "img_index": img_index,
            "cap_index": cap_index,
        }

        return gallery
    
    def faiss_search(self, query_text, gallery, top_k=20):
        q = self.embed_texts(query_text)
        print(q.size())
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
        
    