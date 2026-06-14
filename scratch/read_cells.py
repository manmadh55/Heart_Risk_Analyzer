import json

with open("Heart_disease_risk.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]:
    if idx < len(nb["cells"]):
        cell = nb["cells"][idx]
        print(f"--- Cell {idx} ({cell['cell_type']}) ---")
        print("".join(cell['source']))
