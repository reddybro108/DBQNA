#!/usr/bin/env python3
"""
Convert each .txt file in data/txt into an individual .pdf file in data/pdf.
"""

from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

def txt_to_pdf(txt_path: Path, pdf_path: Path):
    """Convert one .txt to a PDF."""
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    for line in text.splitlines():
        if not line.strip():
            story.append(Spacer(1, 12))
        else:
            story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 6))

    doc.build(story)

def main():
    input_dir = Path("data/txt")
    output_dir = Path("data/pdf")
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list(input_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} .txt files in {input_dir}")

    for txt_file in txt_files:
        pdf_file = output_dir / (txt_file.stem + ".pdf")
        try:
            txt_to_pdf(txt_file, pdf_file)
            print(f"Converted: {txt_file.name} → {pdf_file.name}")
        except Exception as e:
            print(f"Error converting {txt_file.name}: {e}")

if __name__ == "__main__":
    main()
