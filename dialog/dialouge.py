# pip install lmdeploy>=0.7.3
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image

model = 'Qwen/Qwen2-VL-2B-Instruct'

engine_cfg = TurbomindEngineConfig()

template_cfg = ChatTemplateConfig(model_name="qwen2-vl")
pipe = pipeline(model, engine_cfg=engine_cfg, chat_template_config=template_cfg)
history = []

img = load_image("F:\\RAGInteractIR\\datasets\\VG\\VG_100K\\216.jpg")

user = """
You are the Vision-Language Models. 
Please describe this image.
"""
resp = pipe((img, user), history=history)
print(resp.text)