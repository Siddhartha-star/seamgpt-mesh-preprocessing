"""
create_report.py
Simple PDF generator that compiles report_template.md and images into a single PDF.
It requires: markdown, reportlab, pillow
Install: pip install markdown reportlab pillow
"""

import os
import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from markdown import markdown
from PIL import Image

REPORT_TEMPLATE = "report_template.md"
OUTPUT_DIR = "output"
PDF_OUT = "mesh_assignment_report.pdf"

def gather_text_from_template(template_path):
    with open(template_path, "r", encoding="utf-8") as fh:
        md = fh.read()
    # Convert markdown to plain text lines (basic)
    html = markdown(md)
    # strip tags for plain text; simple approach:
    import re
    text = re.sub(r'<[^>]+>', '', html)
    # replace multiple spaces
    text = text.replace('&nbsp;', ' ')
    return text

def add_image_safe(c, img_path, x, y, max_w, max_h):
    try:
        im = Image.open(img_path)
        w, h = im.size
        ratio = min(max_w / w, max_h / h, 1.0)
        w2, h2 = w * ratio, h * ratio
        c.drawInlineImage(img_path, x, y - h2, width=w2, height=h2)
        return h2 + 1*cm
    except Exception as e:
        print("Failed to add image", img_path, e)
        return 0

def main():
    text = gather_text_from_template(REPORT_TEMPLATE)
    c = canvas.Canvas(PDF_OUT, pagesize=A4)
    width, height = A4
    margin = 2*cm
    x = margin
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Mesh Normalization, Quantization & Error Analysis")
    y -= 1*cm

    # Body text - wrap simple
    c.setFont("Helvetica", 10)
    for line in text.splitlines():
        if y < margin + 4*cm:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)
        c.drawString(x, y, line[:1000])
        y -= 0.5*cm

    # Try to include first plot images found under output/*/plots/
    added = 0
    for mesh_dir in sorted(Path(OUTPUT_DIR).iterdir()):
        plots_dir = mesh_dir / "plots"
        if plots_dir.exists():
            for img in sorted(plots_dir.glob("*.png"))[:6]:
                if y < margin + 6*cm:
                    c.showPage()
                    y = height - margin
                c.setFont("Helvetica-Bold", 11)
                c.drawString(x, y, f"Image: {img.name}")
                y -= 0.5*cm
                used_h = add_image_safe(c, str(img), x, y, width - 2*margin, 8*cm)
                y -= used_h + 0.5*cm
                added += 1
                if added > 8:
                    break
    c.save()
    print(f"PDF report generated: {PDF_OUT}")

if __name__ == "__main__":
    main()
