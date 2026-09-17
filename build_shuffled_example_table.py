"""Render the shuffled-label example as a paste-ready raster table."""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "report_assets" / "shuffled_label_example.png"

columns = ["Activation came from", "Correct audience label", "After one shuffle"]
rows = [
    ["ResNet + relevant expert", "Relevant", "Random group +"],
    ["ResNet + unrelated expert", "Unrelated", "Random group −"],
    ["Transformer + relevant expert", "Relevant", "Random group −"],
    ["Transformer + unrelated expert", "Unrelated", "Random group +"],
]

fig, ax = plt.subplots(figsize=(10.2, 2.45))
fig.patch.set_facecolor("white")
ax.axis("off")

table = ax.table(
    cellText=rows,
    colLabels=columns,
    colWidths=[0.43, 0.28, 0.29],
    cellLoc="left",
    colLoc="left",
    bbox=[0.01, 0.03, 0.98, 0.94],
)
table.auto_set_font_size(False)
table.set_fontsize(12)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#CBD5E1")
    cell.set_linewidth(1.0)
    cell.PAD = 0.06
    if row == 0:
        cell.set_facecolor("#17365D")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
    else:
        cell.set_facecolor("#F8FAFC" if row % 2 else "white")
        cell.get_text().set_color("#1F2937")
        if col == 2:
            cell.get_text().set_weight("bold")
            cell.get_text().set_color("#176B87" if rows[row-1][col].endswith("+") else "#B45353")

OUTPUT.parent.mkdir(exist_ok=True)
fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.close(fig)
print(OUTPUT)
