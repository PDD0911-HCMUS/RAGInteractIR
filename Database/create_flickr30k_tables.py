from sqlalchemy import text

from Database.db_session import engine


CREATE_FLICKR30K_TARGET_SQL = """
CREATE TABLE IF NOT EXISTS "Flickr30KTargetAnnotations" (
    "ID" BIGSERIAL PRIMARY KEY,
    "image_id" TEXT NOT NULL,
    "image_path" TEXT NOT NULL,
    "split" TEXT,
    "base_caption" TEXT,
    "captions" JSONB,
    "visual_facts" JSONB,
    "positive_facts" JSONB,
    "negative_facts" JSONB,
    "uncertain_facts" JSONB,
    "entity_phrases" JSONB,
    "grounded_phrases" JSONB,
    "boxes" JSONB,
    "enriched_caption" TEXT,
    "source" TEXT DEFAULT 'flickr30k_entities',
    "created_at" TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_FLICKR30K_CLIP_SQL = """
CREATE TABLE IF NOT EXISTS "Flickr30KCLIPCapDial" (
    "ID" BIGSERIAL PRIMARY KEY,
    "image_id" TEXT,
    "caption" TEXT,
    "captions" JSONB,
    "dialog_id" UUID,
    "img_em" DOUBLE PRECISION[],
    "cap_em" DOUBLE PRECISION[],
    "mode" TEXT,
    "image_path" TEXT,
    "model_name" TEXT,
    "created_at" TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_FLICKR30K_SIGLIP_SQL = """
CREATE TABLE IF NOT EXISTS "Flickr30KSigLIPCapDial" (
    "ID" BIGSERIAL PRIMARY KEY,
    "image_id" TEXT,
    "caption" TEXT,
    "captions" JSONB,
    "dialog_id" UUID,
    "img_em" DOUBLE PRECISION[],
    "cap_em" DOUBLE PRECISION[],
    "mode" TEXT,
    "image_path" TEXT,
    "model_name" TEXT,
    "created_at" TIMESTAMPTZ DEFAULT NOW()
);
"""

INDEX_SQL = [
    'CREATE INDEX IF NOT EXISTS "idx_flickr30k_target_image_id" ON "Flickr30KTargetAnnotations" ("image_id");',
    'CREATE INDEX IF NOT EXISTS "idx_flickr30k_target_image_path" ON "Flickr30KTargetAnnotations" ("image_path");',
    'CREATE INDEX IF NOT EXISTS "idx_flickr30k_target_split" ON "Flickr30KTargetAnnotations" ("split");',
    'CREATE INDEX IF NOT EXISTS "idx_flickr30k_clip_mode_model" ON "Flickr30KCLIPCapDial" ("mode", "model_name");',
    'CREATE INDEX IF NOT EXISTS "idx_flickr30k_clip_image_id" ON "Flickr30KCLIPCapDial" ("image_id");',
    'CREATE INDEX IF NOT EXISTS "idx_flickr30k_clip_image_path" ON "Flickr30KCLIPCapDial" ("image_path");',
    'CREATE INDEX IF NOT EXISTS "idx_flickr30k_siglip_mode_model" ON "Flickr30KSigLIPCapDial" ("mode", "model_name");',
    'CREATE INDEX IF NOT EXISTS "idx_flickr30k_siglip_image_id" ON "Flickr30KSigLIPCapDial" ("image_id");',
    'CREATE INDEX IF NOT EXISTS "idx_flickr30k_siglip_image_path" ON "Flickr30KSigLIPCapDial" ("image_path");',
]


def main() -> None:
    with engine.begin() as conn:
        conn.execute(text(CREATE_FLICKR30K_TARGET_SQL))
        conn.execute(text(CREATE_FLICKR30K_CLIP_SQL))
        conn.execute(text(CREATE_FLICKR30K_SIGLIP_SQL))
        for sql in INDEX_SQL:
            conn.execute(text(sql))
    print("Flickr30K tables are ready.")


if __name__ == "__main__":
    main()
