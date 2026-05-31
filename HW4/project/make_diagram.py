"""Generate architecture diagram as PNG for the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "documents", "architecture.png")

# ── Colors ────────────────────────────────────────────────────────────────────
C_BG      = "#0f1117"
C_CARD    = "#1e2130"
C_ACCENT  = "#6366f1"
C_BORDER  = "#2a2d3a"
C_WHITE   = "#ffffff"
C_LIGHT   = "#e0e0e0"
C_MUTED   = "#6b7280"
C_GREEN   = "#86efac"
C_YELLOW  = "#fde68a"
C_RED     = "#fca5a5"

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")


def box(x, y, w, h, label, sublabel=None,
        fc=C_CARD, ec=C_ACCENT, lw=1.5,
        label_color=C_WHITE, sub_color=C_MUTED,
        label_size=9, sub_size=7.5, radius=0.25):
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           facecolor=fc, edgecolor=ec, linewidth=lw,
                           zorder=2)
    ax.add_patch(rect)
    # top accent bar
    bar = FancyBboxPatch((x, y + h - 0.12), w, 0.12,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=ec, edgecolor="none", zorder=3)
    ax.add_patch(bar)
    cy = y + h / 2 + (0.15 if sublabel else 0)
    ax.text(x + w / 2, cy, label,
            ha="center", va="center",
            color=label_color, fontsize=label_size, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x + w / 2, cy - 0.38, sublabel,
                ha="center", va="center",
                color=sub_color, fontsize=sub_size, zorder=4,
                multialignment="center")


def arrow(x1, y1, x2, y2, label=None, color=C_MUTED, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, connectionstyle="arc3,rad=0"))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.1, my, label, color=color, fontsize=7,
                ha="left", va="center", zorder=5)


def dashed_arrow(x1, y1, x2, y2, label=None, color=C_YELLOW):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=1.5, linestyle="dashed",
                                connectionstyle="arc3,rad=0.0"))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.1, my, label, color=color, fontsize=7,
                ha="left", va="center", zorder=5)


# ── Nodes ─────────────────────────────────────────────────────────────────────

# User question
box(5.5, 8.9, 3.0, 0.7, "User Question", fc=C_BG, ec=C_ACCENT, lw=2,
    label_color=C_ACCENT, label_size=10)

# Orchestrator
box(4.5, 7.2, 5.0, 1.3, "Orchestration Node",
    sublabel="GPT-4o classifier · routes questions\ndrives validator retry loop (max 3)",
    ec=C_ACCENT, lw=2)

# Web search
box(0.4, 4.8, 3.2, 1.8, "DuckDuckGo\nWeb Search",
    sublabel="LangChain integration\nReturns grounded answer\n+ source citations",
    ec="#3b82f6", label_color="#93c5fd")

# Analytics agent
box(4.5, 3.6, 5.0, 3.2, "Enhanced Analytics Agent",
    sublabel="1. Inspect datasets (all columns)\n2. Generate Python code\n3. Execute in persistent sandbox\n4. Observe output\n5. Decide stop / continue\n   (up to 5 iterations)",
    ec=C_GREEN, label_color=C_GREEN, sub_size=7.2)

# Validator
box(10.4, 4.8, 3.2, 1.8, "HW3 Validator Agent",
    sublabel="Reasons about correctness\ncompleteness & consistency\nReturns PASS / RETRY /\nSUSPICIOUS",
    ec=C_YELLOW, label_color=C_YELLOW, sub_size=7)

# Final output
box(4.5, 1.5, 5.0, 1.5, "Final Output",
    sublabel="Natural-language answer\nPlotly JSON visualizations\nCitations (web) / Agent trace (analytics)",
    ec="#10b981", label_color="#6ee7b7", sub_size=7.5)

# ── Arrows ────────────────────────────────────────────────────────────────────

# User → Orchestrator
arrow(7.0, 8.9, 7.0, 8.5, color=C_ACCENT, lw=2)

# Orchestrator → Web Search
arrow(4.5, 7.55, 2.0, 6.6, label="generic\ndomain", color="#3b82f6")

# Orchestrator → Analytics Agent
arrow(7.0, 7.2, 7.0, 6.8, label="analytics", color=C_GREEN, lw=2)

# Analytics Agent → Validator
arrow(9.5, 5.2, 10.4, 5.7, color=C_YELLOW, lw=1.5)

# Validator PASS → Final Output
arrow(12.0, 4.8, 9.0, 3.0, label="PASS", color=C_GREEN)

# Validator RETRY → Orchestrator (dashed)
dashed_arrow(12.0, 6.6, 9.5, 7.55, label="RETRY /\nSUSPICIOUS\n(+feedback)", color=C_YELLOW)

# Web Search → Final Output
arrow(2.0, 4.8, 5.5, 3.0, label="answer +\ncitations", color="#3b82f6")

# Analytics Agent → Final Output (when validator passes)
arrow(7.0, 3.6, 7.0, 3.0, color=C_GREEN, lw=1.5)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (C_ACCENT,  "Orchestration"),
    (C_GREEN,   "Analytics pipeline"),
    ("#3b82f6", "Web search"),
    (C_YELLOW,  "Validation / retry"),
]
for i, (color, label) in enumerate(legend_items):
    ax.add_patch(mpatches.Rectangle((0.3 + i * 3.4, 0.15), 0.25, 0.25,
                                     facecolor=color, edgecolor="none"))
    ax.text(0.65 + i * 3.4, 0.28, label, color=C_MUTED, fontsize=7.5, va="center")

# Title
ax.text(7.0, 9.75, "HW4 System Architecture",
        ha="center", va="center", color=C_WHITE,
        fontsize=13, fontweight="bold")

plt.tight_layout(pad=0.2)
plt.savefig(OUT, dpi=150, facecolor=C_BG, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
