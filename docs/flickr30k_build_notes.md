# Flickr30K Build Notes

This dataset build mirrors the VisDial setup but uses Flickr30K Entities to
derive grounded visual facts.

## Tables

Create all Flickr30K tables:

```bash
python3 Database/create_flickr30k_tables.py
```

The script creates:

- `Flickr30KTargetAnnotations`
- `Flickr30KCLIPCapDial`
- `Flickr30KSigLIPCapDial`

## Expected Dataset Layout

Keep all Flickr-related files under one dataset folder:

```text
datasets/Flickr/
  Images/
    1000092795.jpg
  Entities/
    Annotations/
      1000092795.xml
    Sentences/
      1000092795.txt
    train.txt
    val.txt
    test.txt
```

The DB stores image paths relative to `datasets/Flickr/Images`.

## Build Visual Facts

```bash
python3 datasets/Flickr30KEntities.py \
  --root /workspace/app/datasets/Flickr/Entities \
  --image-folder /workspace/app/datasets/Flickr/Images \
  --split train \
  --skip-existing
```

Quick smoke test:

```bash
python3 datasets/Flickr30KEntities.py \
  --root /workspace/app/datasets/Flickr/Entities \
  --image-folder /workspace/app/datasets/Flickr/Images \
  --split train \
  --limit 10 \
  --skip-existing
```

## Build CLIP Embeddings

```bash
CUDA_VISIBLE_DEVICES=0 python3 datasets/Flickr30KEmbeddings.py \
  --image-folder /workspace/app/datasets/Flickr/Images \
  --backend clip \
  --model-name openai/clip-vit-base-patch32 \
  --mode train \
  --batchsize 512 \
  --device cuda \
  --dtype float32 \
  --caption-mode first \
  --skip-existing
```

## Build SigLIP Embeddings

```bash
CUDA_VISIBLE_DEVICES=0 python3 datasets/Flickr30KEmbeddings.py \
  --image-folder /workspace/app/datasets/Flickr/Images \
  --backend siglip \
  --model-name google/siglip-base-patch16-224 \
  --mode train \
  --batchsize 512 \
  --device cuda \
  --dtype bfloat16 \
  --caption-mode first \
  --skip-existing
```

## Notes

- Visual facts come from Flickr30K Entities sentence phrases and XML bounding
  boxes.
- `grounded_phrases` are phrases linked to boxes.
- `positive_facts` currently equals the selected visual facts.
- `negative_facts` is empty because Flickr30K Entities mostly provides positive
  phrase-region annotations.
