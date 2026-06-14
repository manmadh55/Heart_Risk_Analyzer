import json

with open("Heart_disease_risk.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Number of cells:", len(nb.get("cells", [])))
for i, cell in enumerate(nb.get("cells", [])):
    cell_type = cell.get("cell_type", "")
    source = "".join(cell.get("source", []))
    if cell_type == "markdown":
        print(f"Cell {i} (markdown): {source[:100]}...")
    elif cell_type == "code":
        print(f"Cell {i} (code): {source[:100]}...")
