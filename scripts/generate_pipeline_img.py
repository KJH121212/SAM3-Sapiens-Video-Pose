import matplotlib.patches as patches
import matplotlib.pyplot as plt

def draw_v1_v2_comparison(output_filename="samsapiens_comparison.png"):
    fig, ax = plt.subplots(figsize=(8, 5.0), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    W, H = 3.6, 0.58

    def draw_box(cx, cy, title, subtitle="", color="#ECEFF1", border="#90A4AE"):
        x = cx - W / 2
        y = cy - H / 2
        box = patches.FancyBboxPatch(
            (x, y), W, H,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            fc=color, ec=border, lw=1.2
        )
        ax.add_patch(box)
        if subtitle:
            ax.text(cx, cy + 0.1, title, ha="center", va="center", fontsize=8.5, fontweight="bold")
            ax.text(cx, cy - 0.13, subtitle, ha="center", va="center", fontsize=7.2, color="#555555")
        else:
            ax.text(cx, cy, title, ha="center", va="center", fontsize=8.5, fontweight="bold")

    def draw_arrow(start, end, style="-|>"):
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle=style,
                color="#455A64",
                lw=1.2,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=10
            )
        )

    # Column Headers
    ax.text(2.6, 6.3, "SAMSapiens V1", ha="center", va="center", fontsize=11, fontweight="bold", color="#1565C0")
    ax.text(7.4, 6.3, "SAMSapiens V2", ha="center", va="center", fontsize=11, fontweight="bold", color="#2E7D32")

    # Divider line
    ax.plot([5.0, 5.0], [0.8, 6.5], color="#CFD8DC", linestyle="--", lw=1.2)

    half_h = H / 2

    # --- SAMSapiens V1 Nodes ---
    v1_nodes = {
        "frames":  (2.6, 5.5),
        "sam3":    (1.6, 4.1),
        "sapiens": (3.6, 4.1),
        "assign":  (2.6, 2.7),
        "output":  (2.6, 1.3)
    }

    draw_box(*v1_nodes["frames"], "Frame Extraction", "Input Frames", "#ECEFF1", "#90A4AE")

    # V1 parallel branch boxes (narrower)
    w_sub = 1.7
    for key, (cx, cy), title, sub, bg, bc in [
        ("sam3", v1_nodes["sam3"], "SAM3", "Mask & Track ID", "#E1F5FE", "#0288D1"),
        ("sapiens", v1_nodes["sapiens"], "Sapiens 1", "17 Keypoints", "#FFF3E0", "#FB8C00")
    ]:
        box = patches.FancyBboxPatch(
            (cx - w_sub / 2, cy - H / 2), w_sub, H,
            boxstyle="round,pad=0.06,rounding_size=0.12", fc=bg, ec=bc, lw=1.2
        )
        ax.add_patch(box)
        ax.text(cx, cy + 0.1, title, ha="center", va="center", fontsize=8.2, fontweight="bold")
        ax.text(cx, cy - 0.13, sub, ha="center", va="center", fontsize=6.8, color="#555555")

    draw_box(*v1_nodes["assign"], "Rule-based ID Matching", "Skeleton + Track ID", "#F3E5F5", "#7B1FA2")
    draw_box(*v1_nodes["output"], "V1 Skeleton Output", "17 Keypoints + ID", "#EDE7F6", "#5E35B1")

    # V1 Arrows
    draw_arrow((v1_nodes["frames"][0] - 0.7, v1_nodes["frames"][1] - half_h), (v1_nodes["sam3"][0], v1_nodes["sam3"][1] + half_h))
    draw_arrow((v1_nodes["frames"][0] + 0.7, v1_nodes["frames"][1] - half_h), (v1_nodes["sapiens"][0], v1_nodes["sapiens"][1] + half_h))
    draw_arrow((v1_nodes["sam3"][0], v1_nodes["sam3"][1] - half_h), (v1_nodes["assign"][0] - 0.6, v1_nodes["assign"][1] + half_h))
    draw_arrow((v1_nodes["sapiens"][0], v1_nodes["sapiens"][1] - half_h), (v1_nodes["assign"][0] + 0.6, v1_nodes["assign"][1] + half_h))
    draw_arrow((v1_nodes["assign"][0], v1_nodes["assign"][1] - half_h), (v1_nodes["output"][0], v1_nodes["output"][1] + half_h))

    # --- SAMSapiens V2 Nodes ---
    v2_nodes = {
        "frames":  (7.4, 5.5),
        "sam3":    (7.4, 4.1),
        "sapiens": (7.4, 2.7),
        "output":  (7.4, 1.3)
    }

    draw_box(*v2_nodes["frames"],  "Frame Extraction", "Input Frames", "#ECEFF1", "#90A4AE")
    draw_box(*v2_nodes["sam3"],    "SAM3", "BBox & Mask (Direct Detector)", "#E1F5FE", "#0288D1")
    draw_box(*v2_nodes["sapiens"], "Sapiens 2", "308 Keypoints (from SAM3 BBox)", "#E8F5E9", "#388E3C")
    draw_box(*v2_nodes["output"],  "V2 Skeleton Output", "308 Keypoints (Fingers / Toes)", "#EDE7F6", "#5E35B1")

    # V2 Arrows
    draw_arrow((v2_nodes["frames"][0], v2_nodes["frames"][1] - half_h), (v2_nodes["sam3"][0], v2_nodes["sam3"][1] + half_h))
    draw_arrow((v2_nodes["sam3"][0], v2_nodes["sam3"][1] - half_h), (v2_nodes["sapiens"][0], v2_nodes["sapiens"][1] + half_h))
    draw_arrow((v2_nodes["sapiens"][0], v2_nodes["sapiens"][1] - half_h), (v2_nodes["output"][0], v2_nodes["output"][1] + half_h))

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()
    print(f"Generated: {output_filename}")

if __name__ == "__main__":
    draw_v1_v2_comparison()