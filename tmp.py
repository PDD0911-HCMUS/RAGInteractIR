import json
from pycocotools.coco import COCO
# with open("datasets\\VisDial\\visdial_1.0_train.json", "r") as f:
#     data = json.load(f)
    
# print(data["data"]["dialogs"][0]["image_id"])
# ann_file = "F:\\RAGInteractIR\\datasets\\MSCOCO\\annotations2014\\instances_train2014.json"
# coco = COCO(ann_file)

# info = coco.loadImgs(378466)

# print(info)

# def build_mapping(ann_file: str, split: str) -> dict:
#     coco = COCO(ann_file)
#     mapping = {}
#     for image_id in coco.getImgIds():
#         info = coco.loadImgs([image_id])[0]
#         mapping[str(image_id)] = f"{split}/{info['file_name']}"
#     return mapping

# mapping = {}
# mapping.update(build_mapping("F:\\RAGInteractIR\\datasets\\MSCOCO\\annotations2014\\instances_train2014.json", "train2014"))
# mapping.update(build_mapping("F:\\RAGInteractIR\\datasets\\MSCOCO\\annotations2014\\instances_val2014.json", "val2014"))

# with open("coco2014_id_to_relpath.json", "w", encoding="utf-8") as f:
#     json.dump(mapping, f, ensure_ascii=False, indent=2)

# with open("F:\\RAGInteractIR\\coco2014_id_to_relpath.json", "r") as f:
#     data = json.load(f)

# print(data["378466"])

from openai import OpenAI
client = OpenAI()

text = "dog is laying on bed, it next to a man and a man is watching TV"
convert_triplet = (
            "You are a Vision-Language Model.\n"
            "User question: Convert the given sentence into (subject, relation, object) triplet.\n"
            "Rules:\n"
            "- Do NOT add explanations.\n"
            "- Use lowercase for relation.\n"
            "Return format:\n"
            "[{{\"subject\":\"...\",\"relation\":\"...\",\"object\":\"...\"}}]\n"
            "Sentence:\n"
            "dog is laying on bed, it next to a man and a man is watching TV"
        )

response = client.responses.create(
    model="gpt-5.4",
    input=convert_triplet
)

print(response.output_text)