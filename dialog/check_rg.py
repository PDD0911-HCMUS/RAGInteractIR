import json
import os
from typing import List, Dict

root = "F:\\RAGInteractIR\\datasets\\VG\\"
region_json = "Annotations\\region_descriptions.json"

def extract_top_k_phrases(regions, k=10):
    if not regions:
        return {"image_id": None, "regions": []}

    image_id = regions[0]["image_id"]

    regions_sorted = sorted(
        regions,
        key=lambda r: r["width"] * r["height"],
        reverse=True
    )

    phrases = []
    seen = set()

    for r in regions_sorted:
        phrase = r["phrase"].strip().lower()

        if phrase not in seen:
            phrases.append(phrase)
            seen.add(phrase)

        if len(phrases) >= k:
            break

    return {
        "image_id": image_id,
        "regions": phrases
    }


with open(os.path.join(root, region_json), "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

for item in data:
    top_regions = extract_top_k_phrases(item["regions"], k=10)
    results.append(top_regions)

print(results[0])
print(len(results))

output_path = os.path.join(root, "top_regions.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Saved to:", output_path)