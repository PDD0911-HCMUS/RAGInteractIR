import json

region_json = "F:\\RAGInteractIR\\datasets\\VG\\Annotations\\region_graphs.json"

with open(region_json) as f:
    regions = json.load(f)
    
print(regions[0]["regions"][0].keys())
print(len(regions[0]["regions"][0]))
print(regions[0]["regions"][0]["phrase"])