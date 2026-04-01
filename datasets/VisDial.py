import json
import os
import torch
from transformers import AutoModel, AutoProcessor
import torch.nn.functional as F
from PIL import Image

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from Database.db_session import SessionLocal
from sqlalchemy import select
from Entities.entities import VisDialSigLipCapDial, VisDialSigLipDial, VisDialSigLipAnswers, VisDialSigLipQuestions
import Entities.entities as e
import time
import uuid

'''
- Dữ liệu VisDial sử dụng MSCOCO 2014 cho tập train
- 
'''


class VisDial:
    def __init__(
        self, 
        map_image,
        image_folder, 
        anno, 
        vlm,
        mscoco_ver,
        mode,
        batchsize
        ):
        
        print("[VisDial] Initializing dataset...")
        print(f"[VisDial] anno file      : {anno}")
        print(f"[VisDial] map_image file : {map_image}")
        print(f"[VisDial] image_folder   : {image_folder}")
        print(f"[VisDial] vlm            : {vlm}")
        print(f"[VisDial] mscoco_ver     : {mscoco_ver}")
        print(f"[VisDial] mode           : {mode}")
        print(f"[VisDial] batchsize      : {batchsize}")
        
        with open(anno, 'r') as f:
            self.ann = json.load(f)
        print("[VisDial] Annotation file loaded successfully.")
            
        with open(map_image, 'r') as m:
            self.map = json.load(m)
        print("[VisDial] Image mapping file loaded successfully.")
        
        self.image_folder = image_folder
        self.dialogs = self.ann["data"]["dialogs"]
        self.questions = self.ann["data"]["questions"]
        self.answers = self.ann["data"]["answers"]
        
        self.vlm = vlm
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ver = mscoco_ver
        self.mode = mode
        self.batchsize = batchsize
        
        self.processor, self.model = self.create_vlm()
        
        print(f"[VisDial] Number of dialogs   : {len(self.dialogs)}")
        print(f"[VisDial] Number of questions : {len(self.questions)}")
        print(f"[VisDial] Number of answers   : {len(self.answers)}")
        print(f"[VisDial] Number of map items : {len(self.map)}")
        print(f"[VisDial] Device selected     : {self.device}")
        print("[VisDial] Creating VLM model...")
        pass
    
    def create_vlm(self):
        processor = AutoProcessor.from_pretrained(self.vlm)
        model = AutoModel.from_pretrained(self.vlm).to(self.device).eval()
        
        return processor, model
    
    @torch.no_grad()
    def embed_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        feats = self.model.get_image_features(**inputs)
        feats = F.normalize(feats, dim=-1)

        return feats.squeeze(0).cpu().tolist()
    
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

        return feats.cpu().tolist()
    
    def insert_visdial_questions(self):
        inserted_rows = 0
        buffer = []
        with SessionLocal() as session:
            with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            ) as progress:
                
                task = progress.add_task("[green]Processing insert Questions Table...",total=len(self.questions))

                for item in self.questions:
                    try:
                        q_em = self.embed_texts(item)
                        
                        entity = VisDialSigLipQuestions(
                            question = item,
                            q_em = q_em,
                            mode = self.mode
                        )
                        
                        buffer.append(entity)

                        if len(buffer) >= self.batchsize: # một lần add vào DB một lượng = batchsize
                            session.add_all(buffer)
                            session.commit()

                            inserted_rows += len(buffer)
                            buffer.clear()

                            progress.console.print(
                                f"[cyan]Inserted rows:[/cyan] {inserted_rows}"
                            )

                    except Exception as e:
                        progress.console.print(
                            f"[red]Error: {e}"
                        )

                    progress.advance(task)

            # insert remaining rows
            if buffer:
                session.add_all(buffer)
                session.commit()
                inserted_rows += len(buffer)

        print("\n================================")
        print("Done inserting Questions Table")
        print("Total rows inserted:", inserted_rows)
        print("================================")
                
    def insert_visdial_answers(self):
        inserted_rows = 0
        buffer = []
        with SessionLocal() as session:
            with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            ) as progress:
                
                task = progress.add_task("[green]Processing Answers Table...",total=len(self.answers))

                for item in self.answers:
                    try:
                        ans_em = self.embed_texts(item)
                        
                        entity = VisDialSigLipAnswers(
                            answers = item,
                            ans_em = ans_em,
                            mode = self.mode
                        )
                        
                        buffer.append(entity)

                        if len(buffer) >= self.batchsize: # một lần add vào DB một lượng = batchsize
                            session.add_all(buffer)
                            session.commit()

                            inserted_rows += len(buffer)
                            buffer.clear()

                            progress.console.print(
                                f"[cyan]Inserted rows:[/cyan] {inserted_rows}"
                            )

                    except Exception as e:
                        progress.console.print(
                            f"[red]Error: {e}"
                        )

                    progress.advance(task)

            # insert remaining rows
            if buffer:
                session.add_all(buffer)
                session.commit()
                inserted_rows += len(buffer)

        print("\n================================")
        print("Done inserting Answers Table")
        print("Total rows inserted:", inserted_rows)
        print("================================")
    
    def insert_visdial_cap(self):
        inserted_rows = 0
        buffer = []
        with SessionLocal() as session:
            with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            ) as progress:
                
                task = progress.add_task("[green]Processing Answers Table...",total=len(self.dialogs))   
                for item in self.dialogs:
                    try:
                    
                        image_id = str(item["image_id"])
                        image_path = self.map[image_id]
                        
                        img_em = self.embed_image(os.path.join(self.image_folder, image_path))
                        cap_em = self.embed_texts(item["caption"])
                        
                        dialog_id = uuid.uuid4()
                        
                        vis_cap_dial = VisDialSigLipCapDial(
                            image_id = image_id,
                            caption = item["caption"],
                            dialog_id = dialog_id,
                            img_em = img_em,
                            cap_em = cap_em,
                            mode = self.mode,
                            image_path = image_path
                        )
                        
                        qa_buffer = []
                        for qa in item["dialog"]:
                            ann = qa['answer']
                            q = qa['question']
                            a_opt = qa['answer_options']
                        
                            dial_qa = VisDialSigLipDial(
                                dialog_id = dialog_id,
                                answer = ann,
                                question = q,
                                answer_options = a_opt,
                                mode = self.mode
                            )
                            qa_buffer.append(dial_qa)
                        if(qa_buffer):
                            session.add_all(qa_buffer)
                            session.commit()
                            progress.console.print(
                                f"[cyan]Inserted QA:[/cyan] {dialog_id} with {len(qa_buffer)} QA Done."
                            )
                            
                        buffer.append(vis_cap_dial)
                        if len(buffer) >= self.batchsize: # một lần add vào DB một lượng = batchsize
                            session.add_all(buffer)
                            session.commit()

                            inserted_rows += len(buffer)
                            buffer.clear()

                            progress.console.print(
                                f"[cyan]Inserted rows:[/cyan] {inserted_rows}"
                            )
                    
                    except Exception as e:
                        progress.console.print(
                            f"[red]Error: {e}"
                        )

                    progress.advance(task)
                # insert remaining rows
            if buffer:
                session.add_all(buffer)
                session.commit()
                inserted_rows += len(buffer)

        print("\n================================")
        print("Done inserting Answers Table")
        print("Total rows inserted:", inserted_rows)
        print("================================")
                
    # def build_visdial_db(self):
        
    #     self.insert_visdial_questions()
    #     self.insert_visdial_answers()
    #     self.insert_visdial_cap()
        
if __name__ == "__main__":
    vis_dial = VisDial(
        map_image = "F:\\RAGInteractIR\\datasets\\VisDial\\coco2014_id_to_relpath.json",
        anno = "F:\\RAGInteractIR\\datasets\\VisDial\\visdial_1.0_train.json",
        image_folder = "F:\\RAGInteractIR\\datasets\\MSCOCO",
        vlm = "google/siglip-base-patch16-224",
        mscoco_ver = "2014",
        mode = "train",
        batchsize = 2048
    )
    
    vis_dial.insert_visdial_questions()
    
    vis_dial.insert_visdial_cap()
    
    vis_dial.insert_visdial_answers()